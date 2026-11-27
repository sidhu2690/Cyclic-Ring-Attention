#!/usr/bin/env python3
"""
Experiment 4 (Section VII-D, Table IV):
Complete Expressiveness Battery (408 cells total + Global-Count stress test):
  1. Formal Recognizers (96 cells): 4 tasks (first, parity, cycle3, dyck2) x 4 H x 2 models x 3 seeds
  2. ListOps (24 cells): depth <= 4, T=64, 10-way x 4 H x 2 models x 3 seeds
  3. Length Generalisation (288 cells): train T=32, test T in {32, 64, 128} x 4 tasks x 4 H x 2 models x 3 seeds
  4. Global-Count Probe: 8000-step sum-mod-n check at T=64

Usage:
  python exp4_expressiveness.py --mode all --seeds 0 1 2
  python exp4_expressiveness.py --mode formal --heads 4 8 --seeds 0
"""
import argparse
import random
from typing import Dict, List, Tuple

import torch

from cyclic_core import ExperimentReport, Trainer, build_model, get_device, mean_sd, seed_everything


# =====================================================================
# 1. FORMAL LANGUAGE RECOGNIZERS (first, parity, cycle3, dyck2)
# =====================================================================
def is_valid_dyck2(tokens: List[int]) -> bool:
    """Exact Dyck-2 recognizer over {0:'(', 1:')', 2:'[', 3:']'}."""
    stack = []
    for tok in tokens:
        if tok in (0, 2):
            stack.append(tok)
        elif tok in (1, 3):
            if not stack or stack[-1] != (tok - 1):
                return False
            stack.pop()
        else:
            return False
    return len(stack) == 0


def generate_valid_dyck2_word(seq_len: int, py_rng: random.Random) -> List[int]:
    """Generates a guaranteed balanced Dyck-2 string of even length `seq_len`."""
    assert seq_len % 2 == 0, "Dyck-2 length must be even"
    s, stack = [], []
    for pos in range(seq_len):
        remaining = seq_len - pos
        if len(stack) == remaining:
            # Must close all remaining open brackets to finish balanced at seq_len
            open_tok = stack.pop()
            s.append(open_tok + 1)
        elif not stack or py_rng.random() < 0.55:
            open_tok = py_rng.choice((0, 2))
            stack.append(open_tok)
            s.append(open_tok)
        else:
            open_tok = stack.pop()
            s.append(open_tok + 1)
    return s


class FormalTaskGenerator:
    """
    Binary formal-language recognizers (Table IV, Chance = 0.500 for all four tasks):
      - first:  whether the first symbol is 1 (local regular)
      - parity: whether the total number of 1s is odd (global-count regular)
      - cycle3: whether the sum of increments mod 3 == 0, balanced 50/50 (global-count regular)
      - dyck2:  balanced Dyck-2 strings (y=1) vs random bracket strings (y=0) (context-free)
    """
    def __init__(self, task: str, seq_len: int = 64, vocab: int = 16):
        self.task = task
        self.seq_len = seq_len
        self.vocab = vocab

    def sample_batch(self, batch_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        py_rng = random.Random(seed)
        T = self.seq_len

        if self.task == "first":
            x = torch.randint(0, self.vocab, (batch_size, T), generator=g)
            x[:, 0] = torch.randint(0, 2, (batch_size,), generator=g)
            return x, x[:, 0].clone()

        if self.task == "parity":
            x = torch.randint(0, 2, (batch_size, T), generator=g)
            return x, (x.sum(dim=-1) % 2).long()

        if self.task == "cycle3":
            # Binary membership in {w in {0,1,2}^T : sum(w) == 0 (mod 3)}, balanced 50/50
            xs, ys = [], []
            for i in range(batch_size):
                target_cls = i % 2
                row = [py_rng.randint(0, 2) for _ in range(T - 1)]
                rem = sum(row) % 3
                if target_cls == 1:
                    last = (3 - rem) % 3
                else:
                    last = ((3 - rem) + py_rng.choice((1, 2))) % 3
                row.append(last)
                xs.append(row)
                ys.append(target_cls)
            perm = list(range(batch_size))
            py_rng.shuffle(perm)
            x_t = torch.tensor([xs[p] for p in perm], dtype=torch.long)
            y_t = torch.tensor([ys[p] for p in perm], dtype=torch.long)
            return x_t, y_t

        if self.task == "dyck2":
            xs, ys = [], []
            for i in range(batch_size):
                target_cls = i % 2
                if target_cls == 1:
                    word = generate_valid_dyck2_word(T, py_rng)
                else:
                    word = [py_rng.randint(0, 3) for _ in range(T)]
                    while is_valid_dyck2(word):
                        word = [py_rng.randint(0, 3) for _ in range(T)]
                xs.append(word)
                ys.append(target_cls)
            perm = list(range(batch_size))
            py_rng.shuffle(perm)
            return torch.tensor([xs[p] for p in perm], dtype=torch.long), torch.tensor([ys[p] for p in perm], dtype=torch.long)

        raise ValueError(f"Unknown formal task: {self.task}")

    def balanced_eval_batches(self, n_total: int = 2048, batch_size: int = 512, seed: int = 777):
        x, y = self.sample_batch(n_total, seed=seed)
        return [(x[i : i + batch_size], y[i : i + batch_size]) for i in range(0, n_total, batch_size)]


# =====================================================================
# 2. SCALED-DOWN LISTOPS (Depth <= 4, T=64, 10-way, 204 per class)
# =====================================================================
class ListOpsGenerator:
    OP_MAX, OP_MIN, OP_SUM, CLOSE, PAD = 10, 11, 12, 13, 14
    VOCAB_SIZE = 16
    N_CLASSES = 10

    def __init__(self, seq_len: int = 64, max_depth: int = 4):
        self.seq_len = seq_len
        self.max_depth = max_depth

    def _build_expr(self, depth: int, budget: int, rng: random.Random) -> Tuple[List[int], int]:
        if depth <= 0 or budget < 5:
            val = rng.randint(0, 9)
            return [val], val

        op = rng.choice((self.OP_MAX, self.OP_MIN, self.OP_SUM))
        n_args = rng.choice((2, 3))
        child_budget = max(1, (budget - 2) // n_args)
        tokens = [op]
        vals = []
        for _ in range(n_args):
            sub_depth = depth - 1 if rng.random() < 0.7 else 0
            c_toks, c_val = self._build_expr(sub_depth, child_budget, rng)
            tokens.extend(c_toks)
            vals.append(c_val)
        tokens.append(self.CLOSE)

        if op == self.OP_MAX:
            ans = max(vals)
        elif op == self.OP_MIN:
            ans = min(vals)
        else:
            ans = sum(vals) % 10
        return tokens, ans

    def sample_batch(self, batch_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(seed)
        xs, ys = [], []
        for _ in range(batch_size):
            toks, val = self._build_expr(self.max_depth, self.seq_len - 1, rng)
            toks = toks[: self.seq_len]
            toks += [self.PAD] * (self.seq_len - len(toks))
            xs.append(toks)
            ys.append(val)
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    def balanced_eval_batches(self, per_class: int = 204, batch_size: int = 512, seed: int = 4242):
        rng = random.Random(seed)
        buckets: Dict[int, List[List[int]]] = {c: [] for c in range(self.N_CLASSES)}
        while any(len(v) < per_class for v in buckets.values()):
            toks, val = self._build_expr(self.max_depth, self.seq_len - 1, rng)
            if len(buckets[val]) < per_class:
                toks = toks[: self.seq_len] + [self.PAD] * max(0, self.seq_len - len(toks))
                buckets[val].append(toks)
        xs, ys = [], []
        for c in range(self.N_CLASSES):
            xs.extend(buckets[c])
            ys.extend([c] * len(buckets[c]))
        x_t = torch.tensor(xs, dtype=torch.long)
        y_t = torch.tensor(ys, dtype=torch.long)
        return [(x_t[i : i + batch_size], y_t[i : i + batch_size]) for i in range(0, len(x_t), batch_size)]


# =====================================================================
# 3. RUNNERS FOR TABLE IV, LISTOPS, LENGTH-GEN, & GLOBAL COUNT
# =====================================================================
def run_formal_battery(heads: List[int], seeds: List[int], steps: int, report: ExperimentReport) -> None:
    print("\n" + "=" * 75)
    print("TABLE IV (PART 1): IN-DISTRIBUTION FORMAL BATTERY (T=64, CHANCE=0.5)")
    print("=" * 75)
    device = get_device()
    for task_name in ("first", "parity", "cycle3", "dyck2"):
        gen = FormalTaskGenerator(task_name, seq_len=64, vocab=16)
        for H in heads:
            D = 16 * H
            scores: Dict[str, List[float]] = {"pair": [], "cyclic": []}
            for attn in ("pair", "cyclic"):
                for seed in seeds:
                    seed_everything(seed)
                    eval_set = gen.balanced_eval_batches(n_total=2048, seed=777 + seed)
                    model = build_model(
                        vocab=16, attention=attn, d_model=D, n_heads=H, head_dim=16,
                        ff=D, seq_len=64, n_layers=2, num_classes=2
                    )
                    tr = Trainer(model, lr=1e-3, device=device)
                    for step in range(steps):
                        x, y = gen.sample_batch(128, seed=seed * 100_000 + step)
                        tr.step(x, y)
                    scores[attn].append(tr.evaluate_accuracy(eval_set))

            p_m, p_sd = mean_sd(scores["pair"])
            c_m, c_sd = mean_sd(scores["cyclic"])
            delta = c_m - p_m
            if task_name in ("parity", "cycle3"):
                passed = abs(p_m - 0.5) < 0.06 and abs(c_m - 0.5) < 0.06
                target = "near chance .500"
            else:
                passed = min(p_m, c_m) > 0.90
                target = "learned (> .900)"
            report.add(
                f"Formal {task_name:<6s} H={H:<2d}",
                passed,
                f"pair {p_m:.4f}±{p_sd:.4f} | cyc {c_m:.4f}±{c_sd:.4f} (Δ {delta:+.4f})",
                target,
            )


def run_listops_battery(heads: List[int], seeds: List[int], steps: int, report: ExperimentReport) -> None:
    print("\n" + "=" * 75)
    print("TABLE IV (PART 2): SCALED-DOWN LISTOPS (T=64, DEPTH<=4, CHANCE=0.10)")
    print("=" * 75)
    device = get_device()
    gen = ListOpsGenerator(seq_len=64, max_depth=4)
    for H in heads:
        D = 16 * H
        scores: Dict[str, List[float]] = {"pair": [], "cyclic": []}
        for attn in ("pair", "cyclic"):
            for seed in seeds:
                seed_everything(seed)
                eval_set = gen.balanced_eval_batches(per_class=204, seed=4242 + seed)
                model = build_model(
                    vocab=gen.VOCAB_SIZE, attention=attn, d_model=D, n_heads=H, head_dim=16,
                    ff=D, seq_len=64, n_layers=2, num_classes=10
                )
                tr = Trainer(model, lr=1e-3, device=device)
                for step in range(steps):
                    x, y = gen.sample_batch(128, seed=seed * 100_000 + step)
                    tr.step(x, y)
                scores[attn].append(tr.evaluate_accuracy(eval_set))

        p_m, p_sd = mean_sd(scores["pair"])
        c_m, c_sd = mean_sd(scores["cyclic"])
        report.add(
            f"ListOps H={H:<2d}",
            min(p_m, c_m) > 0.35,
            f"pair {p_m:.3f}±{p_sd:.3f} | cyc {c_m:.3f}±{c_sd:.3f}",
            "above chance 0.10",
        )


def run_length_generalisation(heads: List[int], seeds: List[int], steps: int, report: ExperimentReport) -> None:
    print("\n" + "=" * 75)
    print("SECTION VII-D: LENGTH-GENERALISATION PROBE (288 cells: train T=32,64 -> test T=32,64,128)")
    print("=" * 75)
    device = get_device()
    # 2 train lengths x 2 tasks x 4 heads x 2 models x 3 test lengths x 3 seeds
    # = 288 cells, which is the length-generalisation third of the paper's 408.
    # Training at both 32 and 64 matters: the T=64 run tested at T=32 is the
    # inverse direction and does not always behave like the forward one.
    for train_L in (32, 64):
        for task_name in ("first", "dyck2"):
            train_gen = FormalTaskGenerator(task_name, seq_len=train_L, vocab=16)
            eval_gens = {L: FormalTaskGenerator(task_name, seq_len=L, vocab=16) for L in (32, 64, 128)}
            for H in heads:
                D = 16 * H
                results: Dict[Tuple[str, int], List[float]] = {
                    (a, L): [] for a in ("pair", "cyclic") for L in (32, 64, 128)
                }
                for attn in ("pair", "cyclic"):
                    for seed in seeds:
                        seed_everything(seed)
                        model = build_model(
                            vocab=16, attention=attn, d_model=D, n_heads=H, head_dim=16,
                            ff=D, seq_len=128, n_layers=2, num_classes=2
                        )
                        tr = Trainer(model, lr=1e-3, device=device)
                        for step in range(steps):
                            x, y = train_gen.sample_batch(128, seed=seed * 100_000 + step)
                            tr.step(x, y)
                        for L in (32, 64, 128):
                            eval_set = eval_gens[L].balanced_eval_batches(n_total=1024, seed=909 + seed)
                            results[(attn, L)].append(tr.evaluate_accuracy(eval_set))

                for L in (32, 64, 128):
                    p_m, p_sd = mean_sd(results[("pair", L)])
                    c_m, c_sd = mean_sd(results[("cyclic", L)])
                    is_in_dist = (L == train_L)
                    report.add(
                        f"LengthGen {task_name:<5s} H={H:<2d} tr={train_L}->{L:<3d}",
                        min(p_m, c_m) > 0.90 if is_in_dist else True,
                        f"pair {p_m:.3f}±{p_sd:.3f} | cyc {c_m:.3f}±{c_sd:.3f}",
                        "in-dist > 0.90" if is_in_dist else "OOD probe logged",
                    )


def run_global_count_stress(seeds: List[int], steps: int, report: ExperimentReport) -> None:
    print("\n" + "=" * 75)
    print(f"GLOBAL-COUNT STRESS TEST (sum mod n at T=64, {steps} steps)")
    print("=" * 75)
    device = get_device()
    T, D, H = 64, 128, 4
    for n in (2, 3):
        for attn in ("pair", "cyclic"):
            accs = []
            for seed in seeds:
                seed_everything(seed)
                model = build_model(
                    vocab=n, attention=attn, d_model=D, n_heads=H, head_dim=D // H,
                    ff=2 * D, seq_len=T, n_layers=2, num_classes=n
                )
                tr = Trainer(model, lr=1e-3, device=device)
                g = torch.Generator().manual_seed(seed + 1)
                for _ in range(steps):
                    x = torch.randint(0, n, (128, T), generator=g)
                    y = (x.sum(dim=-1) % n).long()
                    tr.step(x, y)
                ev = torch.Generator().manual_seed(20251126)
                xe = torch.randint(0, n, (4096, T), generator=ev)
                ye = (xe.sum(dim=-1) % n).long()
                eval_batches = [(xe[i : i + 512], ye[i : i + 512]) for i in range(0, 4096, 512)]
                accs.append(tr.evaluate_accuracy(eval_batches))
            m, sd = mean_sd(accs)
            report.add(
                f"sum mod {n} ({attn}, {steps} steps)",
                abs(m - (1.0 / n)) < 0.06,
                f"{m:.4f} ± {sd:.4f}",
                f"near chance {1.0 / n:.3f}",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["formal", "listops", "length_gen", "global_count", "all"], default="all")
    ap.add_argument("--heads", nargs="+", type=int, default=[4, 8, 16, 32])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--global_count_steps", type=int, default=8000)
    ap.add_argument("--out_dir", type=str, default="./results_repro")
    args = ap.parse_args()

    report = ExperimentReport("Section VII-D Expressiveness Battery")
    if args.mode in ("formal", "all"):
        run_formal_battery(args.heads, args.seeds, args.steps, report)
    if args.mode in ("listops", "all"):
        run_listops_battery(args.heads, args.seeds, args.steps, report)
    if args.mode in ("length_gen", "all"):
        run_length_generalisation(args.heads, args.seeds, args.steps, report)
    if args.mode in ("global_count", "all"):
        run_global_count_stress(args.seeds, args.global_count_steps, report)
    report.save("table4_expressiveness.json", out_dir=args.out_dir)


if __name__ == "__main__":
    main()
