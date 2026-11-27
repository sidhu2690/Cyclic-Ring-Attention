#!/usr/bin/env python3
"""
Experiment 3 (Section IV-B, Section VII-C Table III, and Section VIII):
  1. Exact Float64 Verification of the Gradient Identity: W_Q^T G_{W_Q} = G_{W_K}^T W_K (Eq. 10)
  2. Table III: Memory Microbenchmark (H=32, D=512, FF=1024, L=2, BS=16, T=256, Vocab=256),
     including live CUDA peak memory when run on GPU.
  3. Section VIII: 7B-class Model Extrapolation (D=4096, H=32, L=32).

Usage:
  python exp3_memory.py
"""
import argparse
import torch
import torch.nn.functional as F

from cyclic_core import ExperimentReport, Transformer, TransformerConfig, get_device, seed_everything

MIB = 2 ** 20


def verify_gradient_identity_float64(report: ExperimentReport) -> None:
    """Verifies Eq. (10): W_Q^T G_{W_Q} = G_{W_K}^T W_K in float64 to machine precision."""
    seed_everything(0)
    n, d, k = 32, 64, 16
    c = 1.0 / (k ** 0.5)
    X = torch.randn(n, d, dtype=torch.float64)
    WQ = torch.randn(d, k, dtype=torch.float64, requires_grad=True)
    WK = torch.randn(d, k, dtype=torch.float64, requires_grad=True)

    # 1. General upstream gradient G_S
    S = c * (X @ WQ) @ (X @ WK).T
    upstream = torch.randn_like(S)
    loss_gen = (S * upstream).sum()
    loss_gen.backward()
    lhs = WQ.detach().T @ WQ.grad
    rhs = WK.grad.T @ WK.detach()
    # Relative error: the gradient is a sum over n*n=1024 entries, so its entries
    # are O(1e3) and an absolute 1e-12 threshold would fail on scale alone even
    # when the identity holds to machine precision.
    err_gen = float((lhs - rhs).abs().max().item() / lhs.abs().max().item())

    # 2. Cross-entropy loss through softmax attention
    WQ.grad = None
    WK.grad = None
    WV = torch.randn(d, k, dtype=torch.float64)
    S_ce = c * (X @ WQ) @ (X @ WK).T
    A = F.softmax(S_ce, dim=-1)
    out = A @ (X @ WV)
    targets = torch.randint(0, k, (n,))
    loss_ce = F.cross_entropy(out, targets)
    loss_ce.backward()
    lhs_ce = WQ.detach().T @ WQ.grad
    rhs_ce = WK.grad.T @ WK.detach()
    err_ce = float((lhs_ce - rhs_ce).abs().max().item() / lhs_ce.abs().max().item())

    report.add(
        "Eq. (10) identity (general upstream, fp64)",
        err_gen < 1e-12,
        f"max relative diff = {err_gen:.3e}",
        "< 1e-12 (paper quotes 1.4e-14 absolute)",
    )
    report.add(
        "Eq. (10) identity (cross-entropy, fp64)",
        err_ce < 1e-12,
        f"max relative diff = {err_ce:.3e}",
        "< 1e-12 (paper quotes 9.9e-16 absolute)",
    )


def measure_one_step(attention: str, device: torch.device):
    cfg = TransformerConfig(
        vocab=256, d_model=512, n_heads=32, head_dim=16,
        ff=1024, n_layers=2, seq_len=256, attention=attention,
        norm_affine=False, tie_embeddings=False,
    )
    seed_everything(0)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = Transformer(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randint(0, 256, (16, 256), device=device)
    y = torch.randint(0, 256, (16, 256), device=device)

    # Warm-up step: Adam allocates its m and v buffers lazily, inside the first
    # optimizer.step(). Measuring the peak on step 1 therefore excludes the
    # 2 * 4 * N bytes of optimizer state -- which is exactly the quantity this
    # table is about -- and reports a delta that is too small. Do one step, then
    # reset the peak counters with that state already resident.
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()

    total_p = model.total_parameters()
    qk_p = model.qk_parameters()
    peak_mib = (torch.cuda.max_memory_allocated(device) / MIB) if device.type == "cuda" else None
    return total_p, qk_p, peak_mib


def run_table_iii_and_7b(report: ExperimentReport) -> None:
    device = get_device()
    p_tot, p_qk, p_peak = measure_one_step("pair", device)
    c_tot, c_qk, c_peak = measure_one_step("cyclic", device)

    dp_pct = 100.0 * (c_tot - p_tot) / p_tot
    fp32_p_mib, fp32_c_mib = (p_tot * 4) / MIB, (c_tot * 4) / MIB
    adam_p_mib, adam_c_mib = (p_tot * 8) / MIB, (c_tot * 8) / MIB

    report.add("Table III total parameters",
               (p_tot, c_tot) == (5_505_024, 4_980_736),
               f"{p_tot:,} vs {c_tot:,} ({dp_pct:+.1f}%)",
               "5,505,024 vs 4,980,736 (-9.5%)")
    report.add("Table III QK parameters (-50%)",
               (p_qk, c_qk) == (1_048_576, 524_288),
               f"{p_qk:,} vs {c_qk:,}",
               "1,048,576 vs 524,288 (-50.0%)")
    report.add("Table III fp32 params MiB",
               abs((fp32_c_mib - fp32_p_mib) + 2.0) < 0.01,
               f"{fp32_p_mib:.1f} vs {fp32_c_mib:.1f} MiB ({fp32_c_mib - fp32_p_mib:+.1f} MiB)",
               "21.0 vs 19.0 MiB (-2.0 MiB)")
    report.add("Table III Adam m+v MiB",
               abs((adam_c_mib - adam_p_mib) + 4.0) < 0.01,
               f"{adam_p_mib:.1f} vs {adam_c_mib:.1f} MiB ({adam_c_mib - adam_p_mib:+.1f} MiB)",
               "42.0 vs 38.0 MiB (-4.0 MiB)")

    # Peak *allocated* bytes are a property of the allocator, the CUDA version and
    # the kernel choice, not of the architecture, so this row is reported rather
    # than asserted. On the A4000 here it measures -2.00 MiB against the paper's
    # -4.03 MiB: the direction and the order of magnitude agree, the exact delta
    # does not, and the absolute level differs by more than 2x (347 vs 867 MiB)
    # because the peak is dominated by activations at whatever batch the host ran.
    if p_peak is not None and c_peak is not None:
        report.add("Table III CUDA peak allocated MiB", None,
                   f"{p_peak:.2f} vs {c_peak:.2f} MiB ({c_peak - p_peak:+.2f} MiB)",
                   "paper: 867.02 vs 862.99 MiB (-4.03 MiB); allocator-dependent")
    else:
        report.add("Table III CUDA peak allocated MiB", None, "CPU mode", "requires CUDA GPU")

    # Section VIII: 7B Extrapolation (D=4096, H=32, L=32)
    d_7b, l_7b = 4096, 32
    qk_saved = d_7b * d_7b * l_7b
    bytes_saved = 12 * qk_saved  # 4B fp32 param + 8B Adam m,v
    report.add("Section VIII 7B QK params saved",
               abs(qk_saved / 1e9 - 0.537) < 0.01,
               f"{qk_saved:,} ({qk_saved / 1e9:.3f}B)",
               "~0.537B (~half a billion)")
    report.add("Section VIII 7B fp32 + Adam state saved",
               abs(bytes_saved / (2 ** 30) - 6.00) < 0.01,
               f"{bytes_saved / (2 ** 30):.2f} GiB ({bytes_saved / 1e9:.2f} GB)",
               "6.00 GiB (6.44 GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./results_repro")
    args = ap.parse_args()

    print("\n" + "=" * 75)
    print("EXPERIMENT 3: EQ. (10) IDENTITY, TABLE III MEMORY, & 7B EXTRAPOLATION")
    print("=" * 75)
    report = ExperimentReport("Eq. (10), Table III Memory, & 7B Extrapolation")
    verify_gradient_identity_float64(report)
    run_table_iii_and_7b(report)
    report.save("table3_and_7b.json", out_dir=args.out_dir)


if __name__ == "__main__":
    main()
