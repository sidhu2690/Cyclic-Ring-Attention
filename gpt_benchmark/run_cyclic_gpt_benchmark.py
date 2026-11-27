#!/usr/bin/env python3
"""Cross-head cyclic ring vs. normal paired vs. fewer paired heads.

Three architectures, one identical parameter budget per pair of them:

    normal       H=6 paired heads      Q,K,V each D x D    2.D^2 QK params
    cyclic       H=6 ring heads        shared QK, V         1.D^2 QK params
    fewer_heads  H/2=3 paired heads    Q,K over D/2, V      1.D^2 QK params

The baseline that matters is `normal`: the model as anyone would actually write
it, H heads with independent Q and K.

`cyclic` and `fewer_heads` allocate exactly the same parameter tensors
(w_qk, wv, wo of size D x D), so they have identical total and identical QK
parameter counts. The only difference is the wiring: a ring of H shared
projections, or H/2 disjoint (query, key) pairs.

All three are initialised from ONE seeded draw: every model copies the same
base matrix, so two models that own a tensor of the same shape start from
bit-identical values. Comparisons between models are therefore paired, and the
paired t-test in the summary is valid.

    torchrun --nproc_per_node=2 run_cyclic_gpt_benchmark.py \
        --models normal cyclic fewer_heads --seeds 0 1 2 3 4 --max_steps 12000

Changes from the draft this replaces are listed in AUDIT.md.
"""
import argparse
import builtins
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


# =====================================================================
# 1. CONFIGURATION
# =====================================================================
@dataclass
class ExperimentConfig:
    attn_type: Literal["normal", "cyclic", "fewer_heads"] = "cyclic"

    # architecture
    vocab_size: int = 55488          # GPT-2 BPE padded to 57600, rounded up to /64
    n_layer: int = 8
    d_model: int = 384
    n_heads: int = 6
    head_dim: int = 64
    ffn_hidden: int = 1024           # SwiGLU intermediate width
    seq_len: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    # optimisation
    batch_size: int = 64             # global, sequences
    micro_batch_size: int = 24       # per GPU
    max_steps: int = 12000
    eval_interval: int = 500
    eval_batches: int = 50           # per rank
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 400
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 0

    # io
    data_dir: str = "/media/storage/sidhu/cyclic_gpt_scale/data"
    out_dir: str = "./results_cyclic_gpt"


def rms_stable(x, weight, eps):
    """RMSNorm computed in float32 for numerical stability."""
    n = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(n + eps).to(x.dtype))


# =====================================================================
# 2. BUILDING BLOCKS
# =====================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms_stable(x, self.weight, self.eps)


class SplitHalfRoPE(nn.Module):
    """Split-half rotary embedding, applied once per projection (not per head use)."""

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        T = x.size(-2)
        cos = self.cos_cached[:, :, :T].to(x.dtype)
        sin = self.sin_cached[:, :, :T].to(x.dtype)
        return (x * cos) + (self._rotate_half(x) * sin)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# =====================================================================
# 3. THE THREE ATTENTION MODULES
# =====================================================================
class NormalPairedAttention(nn.Module):
    """H independent (query, key) pairs: 2.H matrices of size D x d_h."""


    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.H, self.dh, self.D = cfg.n_heads, cfg.head_dim, cfg.d_model
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.rope = SplitHalfRoPE(cfg.head_dim, cfg.seq_len, cfg.rope_theta)

    def forward(self, x):
        B, T, D = x.shape
        q = self.rope(self.wq(x).view(B, T, self.H, self.dh).transpose(1, 2))
        k = self.rope(self.wk(x).view(B, T, self.H, self.dh).transpose(1, 2))
        v = self.wv(x).view(B, T, self.H, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))


class CyclicRingAttention(nn.Module):
    """H shared projections wired in a ring: S_h uses W_h as query, W_{h+1} as key.

    The projection is computed once, and because keys are literally the queries
    of the neighbouring head, RoPE is applied once to the shared tensor rather
    than twice. That is the whole saving: 1.D^2 instead of 2.D^2 QK parameters.
    """


    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.H, self.dh, self.D = cfg.n_heads, cfg.head_dim, cfg.d_model
        self.w_qk = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.rope = SplitHalfRoPE(cfg.head_dim, cfg.seq_len, cfg.rope_theta)

    def forward(self, x):
        B, T, D = x.shape
        z = self.rope(self.w_qk(x).view(B, T, self.H, self.dh).transpose(1, 2))
        q = z
        k = torch.roll(z, shifts=-1, dims=1)          # head h keys with Z_{h+1}
        v = self.wv(x).view(B, T, self.H, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))


class FewerHeadsPairedAttention(nn.Module):
    """H/2 disjoint pairs (W_0,W_1), (W_2,W_3), ... over the same D x D tensor.

    Identical parameter tensors to CyclicRingAttention, so identical parameter
    counts. H/2 attention maps, each routing 2.d_h value channels.
    """


    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        assert cfg.n_heads % 2 == 0
        self.H, self.dh, self.D = cfg.n_heads, cfg.head_dim, cfg.d_model
        self.Hq = cfg.n_heads // 2
        self.w_qk = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.rope = SplitHalfRoPE(cfg.head_dim, cfg.seq_len, cfg.rope_theta)

    def forward(self, x):
        B, T, D = x.shape
        z = self.rope(self.w_qk(x).view(B, T, self.H, self.dh).transpose(1, 2))
        q = z[:, 0::2]                                  # (B, H/2, T, dh)
        k = z[:, 1::2]
        v = self.wv(x).view(B, T, self.H, self.dh).transpose(1, 2)     # (B,H,T,dh)
        # one map per pair, routing the 2.dh value channels of heads 2m, 2m+1.
        # Split the head axis only: H = (Hq, 2). Reshaping across the time axis
        # here would silently mix positions and break causality.
        v = v.reshape(B, self.Hq, 2, T, self.dh).permute(0, 1, 3, 2, 4)
        v = v.reshape(B, self.Hq, T, 2 * self.dh)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.reshape(B, self.Hq, T, 2, self.dh).permute(0, 1, 3, 2, 4)
        out = out.reshape(B, self.H, T, self.dh).transpose(1, 2)
        return self.wo(out.reshape(B, T, self.H * self.dh))


ATTENTION_REGISTRY = {
    "normal": NormalPairedAttention,
    "cyclic": CyclicRingAttention,
    "fewer_heads": FewerHeadsPairedAttention,
}


# =====================================================================
# 4. BLOCK AND MODEL
# =====================================================================
class TransformerBlock(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = ATTENTION_REGISTRY[cfg.attn_type](cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLUFFN(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LLaMAGPT(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight        # weight tying

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        return logits, loss

    # ---------------------------------------------------------------- accounting
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        qk = 0
        for m in self.modules():
            if isinstance(m, NormalPairedAttention):
                qk += m.wq.weight.numel() + m.wk.weight.numel()
            elif isinstance(m, (CyclicRingAttention, FewerHeadsPairedAttention)):
                qk += m.w_qk.weight.numel()
        return {"total_params": total, "qk_params": qk}



# =====================================================================
# 5. PAIRED INITIALISATION
# =====================================================================
def init_paired_(model: LLaMAGPT, seed: int, verbose: bool = False):
    """Initialise every tensor from (seed, canonical name, shape) only.

    Two properties, both verified in checks.py:

      1. DIFFERENT LAYERS GET DIFFERENT WEIGHTS. The canonical key keeps the
         layer index, so layer i and layer j never share a draw. Dropping the
         index (an earlier version did) aliased all L layers onto one matrix.
      2. DIFFERENT ROLES GET DIFFERENT WEIGHTS. The key is a full SHA-256 of the
         canonical name, so attn.w_qk / attn.wv / attn.wo / ffn.w_gate /
         ffn.w_up / ffn.w_down are independent. Hashing only a prefix of the
         name silently collides ("layers.attn.wq" and "layers.attn.wk" share
         their first 12 characters).

    Cross-model pairing: `wq` is canonicalised to `w_qk`, so the normal model's
    query matrix starts bit-identical to the ring's shared QK matrix. wv, wo,
    the FFN matrices, the embeddings and the norms are shared by name already.
    Cyclic and fewer_heads therefore start from identical weights, and normal
    agrees with them everywhere except that it owns an extra independent `wk`.

    Note the deliberate consequence: all three models now share the same `w_qk`
    draw, so at step 0 the normal model has W_Q = W_K. The first gradient step
    separates them; this is a property of any paired-initialisation design and
    is reported rather than hidden.
    """
    cfg = model.cfg
    n_layer = cfg.n_layer
    cache = {}

    def draw(canonical: str, shape, std: float):
        key = (canonical, tuple(shape), std)
        if key not in cache:
            digest = hashlib.sha256(
                f"seed={seed}|name={canonical}|shape={tuple(shape)}".encode()).digest()
            sub = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
            g = torch.Generator(device="cpu")
            g.manual_seed(sub)
            cache[key] = torch.empty(shape, dtype=torch.float32).normal_(
                mean=0.0, std=std, generator=g)
        return cache[key]

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:                                  # RMSNorm gains
                p.fill_(1.0)
                continue
            if name.endswith("tok_emb.weight") or name.endswith("lm_head.weight"):
                p.copy_(draw("tok_emb.weight", p.shape, 0.02))
                continue
            # keep the layer index; canonicalise only the query role
            canonical = name.replace(".attn.wq.weight", ".attn.w_qk.weight")
            std = 0.02
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                std = 0.02 / math.sqrt(2 * n_layer)
            p.copy_(draw(canonical, p.shape, std))
    if verbose:
        print("[init] paired initialisation applied (layer- and role-aware)", flush=True)


def init_unpaired_(model: LLaMAGPT, seed: int):
    """Reference implementation: one global seed, plain 0.02 normal fill."""
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        if p.dim() == 1:
            nn.init.ones_(p)
        else:
            nn.init.normal_(p, mean=0.0, std=0.02)
    for name, p in model.named_parameters():
        if name.endswith("wo.weight") or name.endswith("w_down.weight"):
            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * model.cfg.n_layer))


# =====================================================================
# 6. DATA
# =====================================================================
class ShardedTokens:
    """Uniform random windows from one or more uint16 token shards."""

    def __init__(self, paths, seq_len, vocab_size):
        self.paths = list(paths)
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.sizes = np.array([a.size for a in self.arrays], dtype=np.int64)
        self.total = int(self.sizes.sum())
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.probs = torch.tensor(self.sizes / self.total, dtype=torch.double)

    def get_batch(self, bs, device, rng=None, start_lo=0, start_hi=None):
        """Windows drawn from [start_lo, start_hi) of the concatenated stream."""
        lo = start_lo
        hi = self.total if start_hi is None else start_hi
        span = hi - lo - self.seq_len - 1
        if span <= 0:
            raise ValueError("requested eval window is too small")
        starts = torch.randint(0, span, (bs,), generator=rng) + lo
        xs = np.empty((bs, self.seq_len + 1), dtype=np.int64)
        for b, s in enumerate(starts.tolist()):
            xs[b] = self._read(s, self.seq_len + 1)
        t = torch.from_numpy(xs)
        # Fail here, loudly, rather than 5000 steps later inside the embedding
        # kernel: an id >= vocab_size is the one way this data pipeline can
        # produce a silent-crash bug, and it costs microseconds to check.
        mx = int(t.max())
        if mx >= self.vocab_size:
            raise ValueError(
                f"token id {mx} >= vocab_size {self.vocab_size}: the vocab was "
                f"detected too small (see detect_vocab_size)")
        return t[:, :-1].to(device, non_blocking=True), t[:, 1:].to(device, non_blocking=True)

    def _read(self, start, n):
        """Read n tokens starting at global index start, across shard boundaries."""
        out = np.empty(n, dtype=np.int64)
        filled = 0
        offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        while filled < n:
            g = start + filled
            i = int(np.searchsorted(offsets, g, side="right") - 1)
            i = min(i, len(self.arrays) - 1)
            local = g - int(offsets[i])
            take = min(n - filled, int(self.sizes[i]) - local)
            out[filled:filled + take] = self.arrays[i][local:local + take]
            filled += take
        return out


def discover_shards(data_dir, split):
    names = sorted(f for f in os.listdir(data_dir)
                   if f.startswith(f"fineweb_{split}") and f.endswith(".bin"))
    if not names:
        raise FileNotFoundError(f"no {split} shards in {data_dir}")
    return [os.path.join(data_dir, n) for n in names]


def detect_vocab_size(paths, chunk=1 << 24, default=None):
    """Exact maximum token id, read in chunks.

    This used to subsample every 7,000,001st token, i.e. about 14 tokens per
    100M-token shard. On this corpus that missed id 57600 and reported 55432,
    which set the vocab to 55488 -- so any token id >= 55488 hit the embedding
    table out of range and CUDA raised "device-side assert triggered" after
    ~5000 steps, killing the whole sweep. Rare ids are exactly the ones a sparse
    sample misses, so the scan has to be exhaustive. 300M uint16 tokens is 600 MB
    of sequential reads, a couple of seconds.
    """
    mx = 0
    for p in paths:
        a = np.memmap(p, dtype=np.uint16, mode="r")
        for lo in range(0, a.size, chunk):
            block = np.asarray(a[lo:lo + chunk])
            if block.size:
                m = int(block.max())
                if m > mx:
                    mx = m
    if default is not None and mx < default:
        mx = default - 1
    padded = ((mx // 64) + 1) * 64
    return padded, mx


# =====================================================================
# 7. TRAINING
# =====================================================================
def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model, ds, cfg, device, rank, world_size, n_batches, lo, hi):
    """Sharded evaluation: each rank scores a different slice of the window."""
    model.eval()
    span = hi - lo
    per = span // world_size
    my_lo = lo + rank * per
    my_hi = hi if rank == world_size - 1 else my_lo + per
    rng = torch.Generator().manual_seed(1234 + rank * 7919)
    tot, cnt = 0.0, 0
    for _ in range(n_batches):
        x, y = ds.get_batch(cfg.micro_batch_size, device, rng=rng,
                            start_lo=my_lo, start_hi=my_hi)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        tot += loss.item() * y.numel()
        cnt += y.numel()
    s = torch.tensor([tot, float(cnt)], device=device)
    if dist.is_initialized():
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
    ce = float(s[0] / s[1])
    model.train()
    return {"ce": ce, "ppl": math.exp(ce)}


def train_single_run(cfg, rank, world_size, device, val_window, test_window):
    is_master = rank == 0
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    model = LLaMAGPT(cfg)
    init_paired_(model, cfg.seed)          # rank-independent, but verify anyway
    model = model.to(device)
    if dist.is_initialized():
        # belt and braces: broadcast rank 0's parameters so every rank provably
        # starts from the same numbers
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
    counts = model.count_parameters()

    ddp = None
    if dist.is_initialized():
        ddp = DDP(model, device_ids=[device.index],
                  broadcast_buffers=False)
    # The training step MUST go through the wrapper: calling the bare module
    # skips DDP.forward, so the reducer never registers the step and gradients
    # are never all-reduced. That silently trains on one GPU's data while
    # paying for two.
    forward_model = ddp if ddp is not None else model

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt_kw = dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8)
    try:
        optim = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": nodecay, "weight_decay": 0.0}], fused=True, **opt_kw)
    except (RuntimeError, TypeError):
        optim = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": nodecay, "weight_decay": 0.0}], **opt_kw)

    train_paths = discover_shards(cfg.data_dir, "train")
    val_paths = discover_shards(cfg.data_dir, "val")
    train_ds = ShardedTokens(train_paths, cfg.seq_len, cfg.vocab_size)
    val_ds = ShardedTokens(val_paths, cfg.seq_len, cfg.vocab_size)

    train_rng = torch.Generator().manual_seed(cfg.seed * 1000 + rank)
    accum = max(1, cfg.batch_size // (cfg.micro_batch_size * world_size))
    effective = cfg.micro_batch_size * world_size * accum
    if effective != cfg.batch_size and is_master:
        print(f"[warn] global batch {cfg.batch_size} is not divisible by "
              f"{cfg.micro_batch_size} x {world_size}; using {effective} instead",
              flush=True)
    tokens_per_step = effective * cfg.seq_len

    if is_master:
        print(f"\n{'='*78}\n"
              f"run: {cfg.attn_type} | seed {cfg.seed} | "
              f"total {counts['total_params']:,} | QK {counts['qk_params']:,} | "
              f"{tokens_per_step:,} tok/step | {accum}x accum\n{'='*78}", flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    history = []
    best = float("inf")
    t_start = time.time()
    t_train = 0.0
    t_eval = 0.0

    for step in range(cfg.max_steps):
        lr = get_lr(step, cfg)
        for g in optim.param_groups:
            g["lr"] = lr

        t0 = time.time()
        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for micro in range(accum):
            if ddp is not None:
                ddp.require_backward_grad_sync = (micro == accum - 1)
            x, y = train_ds.get_batch(cfg.micro_batch_size, device, rng=train_rng)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = forward_model(x, y)
                loss = loss / accum
            loss_accum += loss.item()
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        t_train += time.time() - t0

        if (step + 1) % cfg.eval_interval == 0 or (step + 1) == cfg.max_steps:
            t1 = time.time()
            vm = evaluate(model, val_ds, cfg, device, rank, world_size,
                          cfg.eval_batches, *val_window)
            t_eval += time.time() - t1
            best = min(best, vm["ce"])
            if is_master:
                print(f"step {step+1:6d}/{cfg.max_steps} | train_ce {loss_accum:.4f} | "
                      f"val_ce {vm['ce']:.4f} | val_ppl {vm['ppl']:8.2f} | "
                      f"lr {lr:.2e} | {tokens_per_step/max(t_train/(step+1),1e-9)/1000:6.1f}k tok/s",
                      flush=True)
            history.append({"step": step + 1, "train_ce": loss_accum, **vm})

    final_val = evaluate(model, val_ds, cfg, device, rank, world_size,
                         cfg.eval_batches, *val_window)
    final_test = evaluate(model, val_ds, cfg, device, rank, world_size,
                          cfg.eval_batches, *test_window)
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    wall = time.time() - t_start

    return {
        "attn_type": cfg.attn_type, "seed": cfg.seed,
        "total_params": counts["total_params"], "qk_params": counts["qk_params"],
        "val_ce": final_val["ce"], "val_ppl": final_val["ppl"],
        "best_val_ce": best, "best_val_ppl": math.exp(best),
        "test_ce": final_test["ce"], "test_ppl": final_test["ppl"],
        "peak_vram_mib": peak,
        "tokens_seen": tokens_per_step * cfg.max_steps,
        "train_seconds": t_train, "eval_seconds": t_eval, "wall_seconds": wall,
        "tokens_per_sec_train": tokens_per_step * cfg.max_steps / max(t_train, 1e-9),
        "config": asdict(cfg),
        "history": history,
    }


# =====================================================================
# 8. SUMMARY AND LATEX
# =====================================================================
ORDER = ["normal", "cyclic", "fewer_heads"]
LABEL = {"normal": "Normal paired ($H=6$)",
         "cyclic": "Cyclic ring ($H=6$)",
         "fewer_heads": "Fewer heads ($H/2=3$)"}


def _mean_sd(v):
    if not v:
        return float("nan"), float("nan")
    m = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return m, sd


def _paired_t(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))


def summarize_and_save(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_runs_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    lines = []
    p = lines.append
    p("=" * 100)
    p("FINAL SUMMARY - mean +/- sample SD over seeds (paired initialisation)")
    p("=" * 100)
    p(f"{'config':<12}{'total params':>14}{'QK params':>12}{'val CE':>20}"
      f"{'val PPL':>21}{'test PPL':>21}")
    p("-" * 100)

    latex, by_type = [], {}
    for t in ORDER:
        runs = [r for r in all_results if r["attn_type"] == t]
        if not runs:
            continue
        by_type[t] = {r["seed"]: r for r in runs}
        vce_m, vce_sd = _mean_sd([r["val_ce"] for r in runs])
        vppl_m, vppl_sd = _mean_sd([r["val_ppl"] for r in runs])
        tppl_m, tppl_sd = _mean_sd([r["test_ppl"] for r in runs])
        tps_m, _ = _mean_sd([r["tokens_per_sec_train"] for r in runs])
        p(f"{t:<12}{runs[0]['total_params']:>14,}{runs[0]['qk_params']:>12,}"
          f"{vce_m:>12.4f} ±{vce_sd:.4f}{vppl_m:>12.2f} ±{vppl_sd:.2f}"
          f"{tppl_m:>12.2f} ±{tppl_sd:.2f}   ({tps_m/1000:.1f}k tok/s)")
        latex.append(f"{LABEL[t]} & {runs[0]['total_params']:,} & {runs[0]['qk_params']:,} & "
                     f"${vce_m:.4f} \\pm {vce_sd:.4f}$ & "
                     f"${vppl_m:.2f} \\pm {vppl_sd:.2f}$ & "
                     f"${tppl_m:.2f} \\pm {tppl_sd:.2f}$ \\\\")

    # paired comparisons on identical seeds, identical initial weights
    if "normal" in by_type and len(by_type["normal"]) >= 2:
        p("")
        p("paired comparisons against the normal model (same seeds, same init):")
        p(f"{'contrast':<42}{'delta val CE':>16}{'paired t':>12}{'seeds':>8}")
        for t in ("cyclic", "fewer_heads"):
            if t not in by_type:
                continue
            common = sorted(set(by_type[t]) & set(by_type["normal"]))
            a = [by_type[t][s]["val_ce"] for s in common]
            b = [by_type["normal"][s]["val_ce"] for s in common]
            p(f"{t + ' vs normal':<42}{np.mean(a) - np.mean(b):>+16.4f}"
              f"{_paired_t(a, b):>12.2f}{len(common):>8}")
        if {"cyclic", "fewer_heads"} <= set(by_type):
            common = sorted(set(by_type["cyclic"]) & set(by_type["fewer_heads"]))
            a = [by_type["cyclic"][s]["val_ce"] for s in common]
            b = [by_type["fewer_heads"][s]["val_ce"] for s in common]
            p(f"{'cyclic vs fewer_heads':<42}{np.mean(a) - np.mean(b):>+16.4f}"
              f"{_paired_t(a, b):>12.2f}{len(common):>8}")

    p("")
    p("IEEE table")
    p("")
    p("\\begin{table}[t]")
    p("\\centering")
    n_l = all_results[0]["config"]["n_layer"]
    d_m = all_results[0]["config"]["d_model"]
    n_s = len({r["seed"] for r in all_results})
    p(f"\\caption{{Cross-head cyclic ring vs.\\ paired heads "
      f"($L={n_l}$, $D={d_m}$, $H=6$, {n_s} seeds, mean $\\pm$ sample SD).}}")
    p("\\begin{tabular}{lccccc}")
    p("\\toprule")
    p("Config & Total params & QK params & Val CE & Val PPL & Test PPL \\\\")
    p("\\midrule")
    for row in latex:
        p(row)
    p("\\bottomrule")
    p("\\end{tabular}")
    p("\\end{table}")

    text = "\n".join(lines)
    print("\n" + text + "\n")
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    return text


def plot_curves(all_results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping plot: {e})")
        return
    colors = {"normal": "tab:blue", "cyclic": "tab:orange",
              "fewer_heads": "tab:green"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for t in ORDER:
        runs = [r for r in all_results if r["attn_type"] == t]
        for i, r in enumerate(runs):
            h = r["history"]
            if not h:
                continue
            ax[0].plot([q["step"] for q in h], [q["ce"] for q in h],
                       color=colors[t], lw=1.0, alpha=0.35 if i else 0.95,
                       label=LABEL[t] if i == 0 else None)
        if runs:
            steps = [q["step"] for q in runs[0]["history"]]
            mat = np.array([[q["ce"] for q in r["history"]] for r in runs])
            if mat.ndim == 2 and mat.shape[0] > 1:
                ax[1].plot(steps, mat.mean(0), color=colors[t], lw=2, label=LABEL[t])
                ax[1].fill_between(steps, mat.mean(0) - mat.std(0, ddof=1),
                                   mat.mean(0) + mat.std(0, ddof=1),
                                   color=colors[t], alpha=0.2)
            elif mat.ndim == 2:
                ax[1].plot(steps, mat[0], color=colors[t], lw=2, label=LABEL[t])
    for a, title in zip(ax, ("all seeds", "mean $\\pm$ SD across seeds")):
        a.set_xlabel("step"); a.set_ylabel("held-out cross-entropy")
        a.set_title(title); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "curves.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


# =====================================================================
# 9. ENTRY POINT
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["normal", "cyclic", "fewer_heads"],
                    choices=list(ATTENTION_REGISTRY))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_layer", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--ffn_hidden", type=int, default=1024)
    ap.add_argument("--max_steps", type=int, default=12000)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--micro_batch_size", type=int, default=16,
                    help="per GPU; 16 * 2 GPUs * 2 accum = global batch 64")
    ap.add_argument("--data_dir", default=ExperimentConfig.data_dir)
    ap.add_argument("--out_dir", default=ExperimentConfig.out_dir)
    ap.add_argument("--time_budget_min", type=float, default=0.0,
                    help="0 = run every requested (model, seed) cell")
    ap.add_argument("--dry_run_checks", action="store_true",
                    help="run the correctness checks and exit")
    args = ap.parse_args()

    if args.dry_run_checks:
        import checks
        checks.main()
        return

    distributed = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_master = rank == 0

    train_paths = discover_shards(args.data_dir, "train")
    val_paths = discover_shards(args.data_dir, "val")
    vocab_size, observed_max = detect_vocab_size(train_paths + val_paths)
    if is_master:
        print(f"[data] {len(train_paths)} train shards, {len(val_paths)} val shards, "
              f"observed max token id {observed_max}, padded vocab {vocab_size}",
              flush=True)

    base = ExperimentConfig(
        n_layer=args.n_layer, d_model=args.d_model, n_heads=args.n_heads,
        head_dim=args.d_model // args.n_heads, ffn_hidden=args.ffn_hidden,
        vocab_size=vocab_size, max_steps=args.max_steps,
        eval_interval=args.eval_interval, eval_batches=args.eval_batches,
        seq_len=args.seq_len, batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        data_dir=args.data_dir, out_dir=args.out_dir,
    )

    # held-out windows: validation uses the middle 80% of the val shard,
    # the test window is the final 10% and is touched only at the very end.
    val_total = sum(np.memmap(p, dtype=np.uint16, mode="r").size for p in val_paths)
    val_window = (int(0.05 * val_total), int(0.90 * val_total))
    test_window = (int(0.90 * val_total), val_total)

    results_path = os.path.join(args.out_dir, "all_runs_metrics.json")
    all_results = []
    if os.path.exists(results_path) and is_master:
        try:
            all_results = json.load(open(results_path))
        except Exception:
            all_results = []

    t_begin = time.time()
    for attn_type in args.models:
        for seed in args.seeds:
            done = any(r["attn_type"] == attn_type and r["seed"] == seed
                       for r in all_results)
            if done:
                if is_master:
                    print(f"[skip] {attn_type} seed {seed} already finished")
                continue
            if args.time_budget_min and (time.time() - t_begin) / 60 > args.time_budget_min:
                if is_master:
                    print("[time budget reached, stopping]")
                break
            cfg = ExperimentConfig(**{**asdict(base), "attn_type": attn_type,
                                      "seed": seed})
            res = train_single_run(cfg, rank, world_size, device,
                                   val_window, test_window)
            if is_master:
                all_results.append(res)
                summarize_and_save(all_results, args.out_dir)
                plot_curves(all_results, args.out_dir)
            if distributed:
                dist.barrier()

    if is_master:
        summarize_and_save(all_results, args.out_dir)
        plot_curves(all_results, args.out_dir)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
