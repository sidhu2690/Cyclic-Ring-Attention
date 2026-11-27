#!/usr/bin/env python3
"""
Experiment 1 (Section VII-A):
  1. Fig. 3: Scaled-down MQAR Grokking Trajectory (S=4, V=64, D=64, H=4, 2500 steps)
  2. Table I: Full 8-row MQAR Stress Test across seeds {0, 1, 2}

Usage:
  python exp1_mqar.py --mode both --seeds 0 1 2
  python exp1_mqar.py --mode grokking --seeds 0 1 2
  python exp1_mqar.py --mode table1 --quick
"""
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from cyclic_core import ExperimentReport, Trainer, build_model, get_device, mean_sd, seed_everything


@dataclass
class MQARConfig:
    n_pairs: int = 8            # S key-value pairs
    d_model: int = 64           # D
    n_heads: int = 4            # H
    vocab: int = 256            # V
    batch: int = 128
    steps: int = 8000
    lr: float = 7e-4
    ff: int = 0                 # 0 -> d_model // 2, as in the paper
    head_dim: int = 16
    regime: str = "C"

    @property
    def seq_len(self) -> int:
        return 2 * self.n_pairs + 1

    @property
    def ff_hidden(self) -> int:
        # Must match the paper's MQAR models: ff = d_model // 2 when ff == 0.
        # An earlier version of this file used 2 * d_model, which is 4x wider,
        # so the MQAR cells were measured on a different model than the paper's.
        return self.ff if self.ff > 0 else (self.d_model // 2)

    @property
    def chance(self) -> float:
        return 1.0 / self.vocab

    def label(self) -> str:
        return f"S={self.n_pairs:<2d} D={self.d_model:<3d} H={self.n_heads:<2d}"


class MQARTask:
    """
    Multi-Query Associative Recall: k_1 v_1 k_2 v_2 ... k_S v_S Q -> target v_i where k_i == Q.
    Keys are sampled without replacement per sequence so every key has a unique ground-truth value.
    """
    def __init__(self, cfg: MQARConfig):
        self.cfg = cfg

    def batch(self, batch_size: int, generator: torch.Generator = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, V = batch_size, self.cfg.n_pairs, self.cfg.vocab
        # Sample S distinct keys per sequence using argsort of uniform random scores
        rand_scores = torch.rand((B, V), generator=generator)
        keys = rand_scores.argsort(dim=-1)[:, :S]
        vals = torch.randint(0, V, (B, S), generator=generator)
        pick = torch.randint(0, S, (B, 1), generator=generator)
        query = keys.gather(1, pick)
        target = vals.gather(1, pick).squeeze(1)
        seq = torch.stack([keys, vals], dim=2).reshape(B, 2 * S)
        return torch.cat([seq, query], dim=1), target

    def eval_batches(self, total_examples: int = 1024, batch_size: int = 256, seed: int = 4242):
        g = torch.Generator().manual_seed(seed)
        n_batches = max(1, total_examples // batch_size)
        return [self.batch(batch_size, g) for _ in range(n_batches)]


def table_i_configs() -> List[MQARConfig]:
    grid = [
        (8, 64, 4), (8, 128, 8), (8, 256, 16), (8, 512, 32),
        (10, 64, 4), (10, 128, 8), (10, 256, 16), (10, 512, 32),
    ]
    cfgs = []
    for S, D, H in grid:
        if (S, D, H) == (10, 64, 4):
            cfgs.append(MQARConfig(n_pairs=S, d_model=D, n_heads=H, lr=2e-3, steps=4000, regime="A"))
        else:
            cfgs.append(MQARConfig(n_pairs=S, d_model=D, n_heads=H, lr=7e-4, steps=8000, regime="C"))
    return cfgs


def grokking_config() -> MQARConfig:
    return MQARConfig(n_pairs=4, d_model=64, n_heads=4, vocab=64, batch=64, steps=2500, lr=7e-4, regime="C")


def run_grokking(seeds: List[int], eval_every: int = 100, out_dir: str = "./results_repro") -> Dict:
    print("\n" + "=" * 75)
    print("FIG. 3: MQAR GROKKING TRAJECTORY (S=4, V=64, D=64, H=4, 2500 steps)")
    print("=" * 75)
    cfg = grokking_config()
    task = MQARTask(cfg)
    eval_set = task.eval_batches(total_examples=1024, batch_size=256, seed=4242)
    device = get_device()
    report = ExperimentReport("Fig. 3 MQAR Grokking")

    trajectories: Dict[str, List[List[Tuple[int, float]]]] = {"pair": [], "cyclic": []}
    cross_steps: Dict[str, List[int]] = {"pair": [], "cyclic": []}

    for attn in ("pair", "cyclic"):
        for seed in seeds:
            seed_everything(seed)
            model = build_model(
                vocab=cfg.vocab, attention=attn, d_model=cfg.d_model,
                n_heads=cfg.n_heads, head_dim=cfg.head_dim, ff=cfg.ff_hidden,
                seq_len=cfg.seq_len, n_layers=2
            )
            tr = Trainer(model, lr=cfg.lr, device=device)
            g = torch.Generator().manual_seed(seed * 1000 + 17)
            curve = []
            crossed = None
            for step in range(cfg.steps):
                x, y = task.batch(cfg.batch, g)
                tr.step(x, y)
                if (step + 1) % eval_every == 0 or (step + 1) == cfg.steps:
                    acc = tr.evaluate_accuracy(eval_set)
                    curve.append((step + 1, acc))
                    if crossed is None and acc >= 0.90:
                        crossed = step + 1
            trajectories[attn].append(curve)
            if crossed is not None:
                cross_steps[attn].append(crossed)
            print(f"  [{attn:<6s} | seed {seed}] start={curve[0][1]:.3f} -> final={curve[-1][1]:.3f} (crossed 0.90 at step {crossed})")

        starts = [c[0][1] for c in trajectories[attn]]
        finals = [c[-1][1] for c in trajectories[attn]]
        report.add(
            f"Fig. 3 {attn}: rises from low acc to > 0.90",
            np.mean(starts) < 0.45 and np.mean(finals) > 0.90,
            f"mean start {np.mean(starts):.3f} -> final {np.mean(finals):.3f}",
            "start < 0.45 -> final > 0.90",
        )

    # Save optional plot matching Fig. 3
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        colors = {"pair": "tab:blue", "cyclic": "tab:teal"}
        for attn in ("pair", "cyclic"):
            steps = [pt[0] for pt in trajectories[attn][0]]
            mat = np.array([[pt[1] for pt in curve] for curve in trajectories[attn]])
            for row in mat:
                ax.plot(steps, row, color=colors[attn], alpha=0.3, lw=1.0)
            ax.plot(steps, mat.mean(axis=0), color=colors[attn], lw=2.2, label=attn)
        ax.axhline(1.0 / cfg.vocab, color="tab:red", ls="--", lw=1.0, label=f"chance = 1/{cfg.vocab}")
        ax.set_xlabel("training steps")
        ax.set_ylabel("MQAR accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"MQAR S=4, V=64, D=64, H=4 ({len(seeds)} seeds, mean bold)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig3_grokking.png"), dpi=160)
        plt.close(fig)
    except Exception:
        pass

    report.save("fig3_grokking.json", out_dir=out_dir)
    return trajectories


def run_table_i(seeds: List[int], quick: bool = False, out_dir: str = "./results_repro") -> None:
    print("\n" + "=" * 75)
    print("TABLE I: MQAR RESULTS BY CONFIGURATION")
    print("=" * 75)
    rows = table_i_configs()[:2] if quick else table_i_configs()
    device = get_device()
    report = ExperimentReport("Table I MQAR Stress Test")

    for row in rows:
        task = MQARTask(row)
        eval_set = task.eval_batches(total_examples=1024, batch_size=256, seed=4242)
        scores: Dict[str, List[float]] = {"pair": [], "cyclic": []}
        for attn in ("pair", "cyclic"):
            for seed in seeds:
                seed_everything(seed)
                model = build_model(
                    vocab=row.vocab, attention=attn, d_model=row.d_model,
                    n_heads=row.n_heads, head_dim=row.head_dim, ff=row.ff_hidden,
                    seq_len=row.seq_len, n_layers=2
                )
                tr = Trainer(model, lr=row.lr, device=device)
                g = torch.Generator().manual_seed(seed * 1000 + 31)
                for _ in range(row.steps):
                    x, y = task.batch(row.batch, g)
                    tr.step(x, y)
                acc = tr.evaluate_accuracy(eval_set)
                scores[attn].append(acc)

        p_m, p_sd = mean_sd(scores["pair"])
        c_m, c_sd = mean_sd(scores["cyclic"])
        delta = c_m - p_m
        report.add(
            f"Table I {row.label()} (regime {row.regime})",
            min(p_m, c_m) > 0.80,
            f"pair {p_m:.3f}±{p_sd:.3f} | cyc {c_m:.3f}±{c_sd:.3f} (Δ {delta:+.3f})",
            "both > 0.80 (chance=0.004)",
            pair_scores=scores["pair"],
            cyclic_scores=scores["cyclic"],
        )

    report.save("table1_mqar.json", out_dir=out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grokking", "table1", "both"], default="both")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--quick", action="store_true", help="Run first 2 rows of Table I for a fast check")
    ap.add_argument("--out_dir", type=str, default="./results_repro")
    args = ap.parse_args()

    if args.mode in ("grokking", "both"):
        run_grokking(args.seeds, out_dir=args.out_dir)
    if args.mode in ("table1", "both"):
        run_table_i(args.seeds, quick=args.quick, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
