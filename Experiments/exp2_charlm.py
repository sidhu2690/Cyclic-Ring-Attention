#!/usr/bin/env python3
"""
Experiment 2 (Section VII-B, Table II):
Character-Level Language Model (276 KB corpus, vocab 146, 2-layer, T=64, BS=32, 2500 steps).
Verifies exact parameter counts for all 5 rows of Table II and reports held-out Val CE across seeds.

Usage:
  python exp2_charlm.py --seeds 0 1 2
  python exp2_charlm.py --text_path /path/to/corpus.txt --seeds 0 1 2
"""
import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from cyclic_core import ExperimentReport, Trainer, build_model, get_device, mean_sd, seed_everything

PAPER_VOCAB = 146
PAPER_BYTES = 276 * 1024


@dataclass
class CharLMConfig:
    label: str
    attention: str
    n_heads: int
    d_model: int
    ff: int
    expected_params: int
    seq_len: int = 64
    batch: int = 32
    steps: int = 2500
    lr: float = 3e-4
    n_layers: int = 2
    head_dim: int = 16
    corpus_bytes: int = PAPER_BYTES
    holdout_frac: float = 0.05


def table_ii_configs() -> List[CharLMConfig]:
    return [
        CharLMConfig("16 pair",         "pair",   n_heads=16, d_model=256, ff=512,  expected_params=1_385_472),
        CharLMConfig("16 cyclic",       "cyclic", n_heads=16, d_model=256, ff=512,  expected_params=1_254_400),
        CharLMConfig("16 cyclic FF600*","cyclic", n_heads=16, d_model=256, ff=600,  expected_params=1_389_568),
        CharLMConfig("32 pair",         "pair",   n_heads=32, d_model=512, ff=1024, expected_params=5_392_384),
        CharLMConfig("32 cyclic",       "cyclic", n_heads=32, d_model=512, ff=1024, expected_params=4_868_096),
    ]


def load_or_build_corpus(text_path: str = "", n_bytes: int = PAPER_BYTES, seed: int = 0) -> Tuple[str, List[str], bool]:
    if text_path and os.path.exists(text_path):
        raw = open(text_path, encoding="utf-8", errors="ignore").read()
        freq: Dict[str, int] = {}
        for ch in raw:
            freq[ch] = freq.get(ch, 0) + 1
        kept = set(sorted(freq, key=freq.get, reverse=True)[:PAPER_VOCAB])
        filtered = "".join(c for c in raw if c in kept)[:n_bytes]
        alphabet = sorted(kept)
        return filtered, alphabet, False

    # Deterministic 146-character structured Markov corpus when real text is not supplied
    alphabet = [chr(c) for c in range(32, 127)]
    extra = [chr(c) for c in range(160, 160 + (PAPER_VOCAB - len(alphabet)))]
    chars = alphabet + extra
    rng = random.Random(seed)
    transitions = {c: rng.sample(chars, 8) for c in chars}
    out, cur = [], chars[0]
    for _ in range(n_bytes):
        cur = rng.choice(transitions[cur])
        out.append(cur)
    return "".join(out), chars, True


class CharLMTask:
    def __init__(self, text_path: str = "", seq_len: int = 64, holdout_frac: float = 0.05):
        self.seq_len = seq_len
        self.text, self.alphabet, self.is_synthetic = load_or_build_corpus(text_path)
        self.stoi = {c: i for i, c in enumerate(self.alphabet)}
        self.data = torch.tensor([self.stoi[c] for c in self.text], dtype=torch.long)
        self.cut = int(len(self.data) * (1.0 - holdout_frac))

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    def sample_batch(self, batch_size: int, generator: torch.Generator, split: str = "train"):
        T = self.seq_len
        lo, hi = (0, self.cut - T - 1) if split == "train" else (self.cut, len(self.data) - T - 1)
        idx = torch.randint(lo, hi, (batch_size,), generator=generator)
        x = torch.stack([self.data[i : i + T] for i in idx])
        y = torch.stack([self.data[i + 1 : i + 1 + T] for i in idx])
        return x, y

    def eval_batches(self, batch_size: int = 32, n_batches: int = 50, seed: int = 1234):
        g = torch.Generator().manual_seed(seed)
        return [self.sample_batch(batch_size, g, split="val") for _ in range(n_batches)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_path", type=str, default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--out_dir", type=str, default="./results_repro")
    args = ap.parse_args()

    print("\n" + "=" * 75)
    print("TABLE II: CHARACTER-LEVEL LANGUAGE MODEL")
    print("=" * 75)
    task = CharLMTask(text_path=args.text_path)
    eval_set = task.eval_batches(batch_size=32, n_batches=50, seed=1234)
    device = get_device()
    report = ExperimentReport("Table II Character-Level LM")

    for cfg in table_ii_configs():
        ces = []
        actual_params = 0
        for seed in args.seeds:
            seed_everything(seed)
            model = build_model(
                vocab=task.vocab_size, attention=cfg.attention, d_model=cfg.d_model,
                n_heads=cfg.n_heads, head_dim=cfg.head_dim, ff=cfg.ff,
                seq_len=cfg.seq_len, n_layers=cfg.n_layers
            )
            actual_params = model.total_parameters()
            tr = Trainer(model, lr=cfg.lr, device=device)
            g = torch.Generator().manual_seed(seed * 1000 + 7)
            for _ in range(args.steps):
                x, y = task.sample_batch(cfg.batch, g, split="train")
                tr.step(x, y)
            val_ce = tr.evaluate_ce(eval_set)
            ces.append(val_ce)

        ce_m, ce_sd = mean_sd(ces)
        report.add(
            f"{cfg.label:<17s} param count",
            actual_params == cfg.expected_params,
            f"{actual_params:,}",
            f"{cfg.expected_params:,}",
        )
        report.add(
            f"{cfg.label:<17s} held-out Val CE",
            True,
            f"{ce_m:.4f} ± {ce_sd:.4f}",
            "logged (lower is better)",
            seed_ces=ces,
        )

    report.save("table2_charlm.json", out_dir=args.out_dir)


if __name__ == "__main__":
    main()
