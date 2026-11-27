#!/usr/bin/env python3
"""
Core models, attention wirings, trainer, and reporting utilities for:
"Your Query Is My Key: Halving Attention Parameters via Cross-Head Cyclic Rings"
"""
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. REPRODUCIBILITY & DEVICE UTILITIES
# =====================================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 2. BUILDING BLOCKS: SplitHalfRoPE, RMSNorm, SwiGLU
# =====================================================================
class SplitHalfRoPE(nn.Module):
    """Split-half Rotary Position Embedding (Su et al., 2021)."""
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for SplitHalfRoPE, got {head_dim}")
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, dh)
        T = x.size(-2)
        cos = self.cos_cached[:, :, :T, :].to(x.dtype)
        sin = self.sin_cached[:, :, :T, :].to(x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)


class RMSNorm(nn.Module):
    """
    Root-Mean-Square Layer Normalization (Zhang & Sennrich, 2020).
    affine=False by default so parameter counts match Tables II and III to the last digit.
    """
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(norm + self.eps).to(x.dtype)
        return out if self.weight is None else out * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020)."""
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# =====================================================================
# 3. ATTENTION WIRINGS: Pair, Cyclic Ring, and Fewer-Heads Paired
# =====================================================================
class PairAttention(nn.Module):
    """Standard multi-head attention with independent Q and K projections (2 * D^2 QK params)."""
    kind = "pair"

    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 seq_len: int = 2048, rope_theta: float = 10000.0):
        super().__init__()
        assert n_heads * head_dim == d_model, "n_heads * head_dim must equal d_model"
        self.d_model, self.H, self.dh = d_model, n_heads, head_dim
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.rope = SplitHalfRoPE(head_dim, max_seq_len=max(seq_len, 2048), theta=rope_theta)

    def qk_parameters(self) -> int:
        return self.wq.weight.numel() + self.wk.weight.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.rope(self.wq(x).view(B, T, self.H, self.dh).transpose(1, 2))
        k = self.rope(self.wk(x).view(B, T, self.H, self.dh).transpose(1, 2))
        v = self.wv(x).view(B, T, self.H, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))


class CyclicAttention(nn.Module):
    """
    Cross-head cyclic ring (Section V, Eq. 11):
      S_h = (1 / sqrt(d_h)) * R(X W_h) R(X W_{(h+1) mod H})^T
    Uses 1 projection per head (1 * D^2 QK params, -50% vs PairAttention).
    """
    kind = "cyclic"

    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 seq_len: int = 2048, rope_theta: float = 10000.0):
        super().__init__()
        assert n_heads * head_dim == d_model, "n_heads * head_dim must equal d_model"
        self.d_model, self.H, self.dh = d_model, n_heads, head_dim
        self.w_qk = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.rope = SplitHalfRoPE(head_dim, max_seq_len=max(seq_len, 2048), theta=rope_theta)

    def qk_parameters(self) -> int:
        return self.w_qk.weight.numel()

    def head_queries(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        z = self.w_qk(x).view(B, T, self.H, self.dh).transpose(1, 2)
        return self.rope(z)

    @staticmethod
    def keys_from_queries(z: torch.Tensor) -> torch.Tensor:
        return torch.roll(z, shifts=-1, dims=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        z = self.head_queries(x)
        k = self.keys_from_queries(z)
        v = self.wv(x).view(B, T, self.H, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(z, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))


class FewerHeadsAttention(nn.Module):
    """
    H/2 independent paired heads over the same 1 * D^2 QK budget as CyclicAttention.
    Pairs (W_{2m}, W_{2m+1}) form H/2 attention maps, each routing 2 * d_h value channels.
    """
    kind = "fewer_heads"

    def __init__(self, d_model: int, n_heads: int, head_dim: int,
                 seq_len: int = 2048, rope_theta: float = 10000.0):
        super().__init__()
        assert n_heads * head_dim == d_model and n_heads % 2 == 0
        self.d_model, self.H, self.dh = d_model, n_heads, head_dim
        self.Hq = n_heads // 2
        self.w_qk = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.rope = SplitHalfRoPE(head_dim, max_seq_len=max(seq_len, 2048), theta=rope_theta)

    def qk_parameters(self) -> int:
        return self.w_qk.weight.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        z = self.rope(self.w_qk(x).view(B, T, self.H, self.dh).transpose(1, 2))
        q = z[:, 0::2]  # (B, H/2, T, dh)
        k = z[:, 1::2]  # (B, H/2, T, dh)
        v = self.wv(x).view(B, T, self.H, self.dh).transpose(1, 2)
        v = v.reshape(B, self.Hq, 2, T, self.dh).permute(0, 1, 3, 2, 4).reshape(B, self.Hq, T, 2 * self.dh)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.reshape(B, self.Hq, T, 2, self.dh).permute(0, 1, 3, 2, 4).reshape(B, self.H, T, self.dh)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))


ATTENTION_REGISTRY = {
    "pair": PairAttention,
    "normal": PairAttention,
    "cyclic": CyclicAttention,
    "fewer_heads": FewerHeadsAttention,
}


# =====================================================================
# 4. TRANSFORMER MODEL
# =====================================================================
@dataclass
class TransformerConfig:
    vocab: int = 256
    d_model: int = 64
    n_heads: int = 4
    head_dim: int = 16
    ff: int = 128
    n_layers: int = 2
    seq_len: int = 256
    rope_theta: float = 10000.0
    attention: str = "pair"
    rope_max_len: int = 2048
    norm_affine: bool = False
    tie_embeddings: bool = False
    num_classes: Optional[int] = None  # If None, outputs `vocab` logits

    def __post_init__(self):
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(f"H * dh must equal d_model ({self.n_heads} * {self.head_dim} != {self.d_model})")


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, affine=cfg.norm_affine)
        attn_cls = ATTENTION_REGISTRY[cfg.attention]
        self.attn = attn_cls(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            seq_len=max(cfg.seq_len, cfg.rope_max_len),
            rope_theta=cfg.rope_theta,
        )
        self.ff_norm = RMSNorm(cfg.d_model, affine=cfg.norm_affine)
        self.ff = SwiGLU(cfg.d_model, cfg.ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        out_dim = cfg.num_classes if cfg.num_classes is not None else cfg.vocab
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, affine=cfg.norm_affine)
        self.head = nn.Linear(cfg.d_model, out_dim, bias=False)
        if cfg.tie_embeddings and out_dim == cfg.vocab:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def qk_parameters(self) -> int:
        return sum(b.attn.qk_parameters() for b in self.blocks)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.norm_f(x))
        loss = None
        if targets is not None:
            if targets.dim() == 1:
                loss = F.cross_entropy(logits[:, -1, :], targets)
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss


def build_model(
    vocab: int,
    attention: str,
    d_model: int,
    n_heads: int,
    head_dim: int,
    ff: int,
    seq_len: int,
    n_layers: int = 2,
    num_classes: Optional[int] = None,
    norm_affine: bool = False,
    tie_embeddings: bool = False,
) -> Transformer:
    cfg = TransformerConfig(
        vocab=vocab,
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        ff=ff,
        n_layers=n_layers,
        seq_len=seq_len,
        attention=attention,
        num_classes=num_classes,
        norm_affine=norm_affine,
        tie_embeddings=tie_embeddings,
    )
    return Transformer(cfg)


# =====================================================================
# 5. TRAINER (Supports Sequence Classification & Autoregressive LM)
# =====================================================================
class Trainer:
    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.0,
                 device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def step(self, x: torch.Tensor, y: torch.Tensor,
             loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> float:
        self.model.train()
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        if loss_fn is not None:
            logits, _ = self.model(x)
            loss = loss_fn(logits, y)
        else:
            _, loss = self.model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate_accuracy(self, batches: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Evaluates final-token classification accuracy across batches."""
        self.model.eval()
        correct, total = 0, 0
        for x, y in batches:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits, _ = self.model(x)
            preds = logits[:, -1, :].argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
        self.model.train()
        return correct / max(1, total)

    @torch.no_grad()
    def evaluate_ce(self, batches: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Evaluates token-level cross-entropy loss across batches."""
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for x, y in batches:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            _, loss = self.model(x, y)
            n_tok = int(y.numel())
            total_loss += float(loss.item()) * n_tok
            total_tokens += n_tok
        self.model.train()
        return total_loss / max(1, total_tokens)


# =====================================================================
# 6. REPORTING & SUMMARY HELPERS
# =====================================================================
def mean_sd(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    m = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return m, sd


class ExperimentReport:
    def __init__(self, title: str):
        self.title = title
        self.records: List[Dict] = []

    def add(self, name: str, passed: Optional[bool], measured: str, expected: str, **extra) -> None:
        status = "INFO" if passed is None else ("PASS" if passed else "FAIL")
        rec = {"name": name, "status": status, "measured": measured, "expected": expected, **extra}
        self.records.append(rec)
        print(f"  [{status:4s}] {name:<46} | measured: {measured:<26} | target: {expected}")

    def save(self, filename: str, out_dir: str = "./results_repro") -> int:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, "w") as f:
            json.dump({"title": self.title, "records": self.records}, f, indent=2)
        n_fail = sum(1 for r in self.records if r["status"] == "FAIL")
        print(f"\nSaved report -> {path} ({len(self.records)} checks, {n_fail} failed)\n")
        return 1 if n_fail > 0 else 0
