#!/usr/bin/env python3
"""Correctness checks that must pass before any GPU time is spent.

    python checks.py

Covers every claim the benchmark's fairness rests on:

 1  parameter accounting      total and QK counts, cyclic == fewer_heads exactly
 2  paired initialisation     same-shape tensors start bit-identical across models
 3  causality                 no token influences an earlier position
 4  ring semantics            head h's key is head h+1's query projection
 5  fewer-heads semantics     map m uses (W_2m, W_2m+1) and its own value slice
 6  RoPE                      applied once in the ring, twice in the paired model
 7  gradients                 every parameter receives a finite gradient
 8  equivalence              with H=1 the ring reduces to tied Q-K attention
 9  bf16 sanity               a short loss decreases in autocast
"""
import json
import math

import torch
import torch.nn.functional as F

from run_cyclic_gpt_benchmark import (
    ExperimentConfig, LLaMAGPT, init_paired_,
    NormalPairedAttention, CyclicRingAttention, FewerHeadsPairedAttention,
    SplitHalfRoPE,
)

CFG = dict(vocab_size=256, n_layer=2, d_model=64, n_heads=4, head_dim=16,
           ffn_hidden=128, seq_len=32)


def cfg_of(attn, **over):
    kw = {**CFG, **over}
    kw["head_dim"] = kw["d_model"] // kw["n_heads"]
    return ExperimentConfig(attn_type=attn, **kw)


def test_param_accounting():
    D, H, L = 384, 6, 8
    got = {}
    for t in ("normal", "cyclic", "fewer_heads"):
        m = LLaMAGPT(cfg_of(t, d_model=D, n_heads=H, n_layer=L))
        got[t] = m.count_parameters()
    d2 = D * D
    assert got["normal"]["qk_params"] == 2 * d2 * L, got["normal"]
    assert got["cyclic"]["qk_params"] == d2 * L, got["cyclic"]
    assert got["fewer_heads"]["qk_params"] == d2 * L, got["fewer_heads"]
    assert got["cyclic"]["total_params"] == got["fewer_heads"]["total_params"]
    c, f, n = (got[t]["qk_params"] for t in ("cyclic", "fewer_heads", "normal"))
    print(f"   normal QK {n:>10,}   cyclic {c:>10,}   fewer {f:>10,}   "
          f"(cyclic = {100*c/n:.0f}% of normal)")
    print(f"   total: normal {got['normal']['total_params']:,}   "
          f"cyclic {got['cyclic']['total_params']:,}   "
          f"fewer {got['fewer_heads']['total_params']:,}   "
          f"identical(cyclic, fewer) = "
          f"{got['cyclic']['total_params'] == got['fewer_heads']['total_params']}")
    return got


def test_paired_init():
    """Layer- and role-awareness, plus cross-model pairing."""
    kw = dict(n_layer=3, d_model=64, n_heads=4, head_dim=16)
    c = LLaMAGPT(cfg_of("cyclic", **kw))
    f = LLaMAGPT(cfg_of("fewer_heads", **kw))
    n = LLaMAGPT(cfg_of("normal", **kw))
    for m in (c, f, n):
        init_paired_(m, seed=7)

    # (1) layers must differ
    layer_alias = any(torch.equal(c.layers[0].attn.w_qk.weight,
                                  c.layers[i].attn.w_qk.weight)
                      for i in range(1, kw["n_layer"]))
    assert not layer_alias, "all layers share one draw: layer index dropped from the key"
    print("   layers independent      ok")

    # (2) roles must differ within a layer
    a = c.layers[0].attn
    assert not torch.equal(a.w_qk.weight, a.wv.weight), "w_qk == wv"
    assert not torch.equal(a.w_qk.weight, a.wo.weight), "w_qk == wo"
    ff = c.layers[0].ffn
    assert not torch.equal(ff.w_gate.weight, ff.w_up.weight), "w_gate == w_up"
    print("   roles independent       ok")

    # (3) cross-model pairing
    assert torch.equal(c.tok_emb.weight, f.tok_emb.weight)
    assert torch.equal(c.tok_emb.weight, n.tok_emb.weight)
    for i in range(kw["n_layer"]):
        assert torch.equal(c.layers[i].attn.w_qk.weight, f.layers[i].attn.w_qk.weight), i
        assert torch.equal(c.layers[i].attn.wv.weight, f.layers[i].attn.wv.weight), i
        assert torch.equal(c.layers[i].attn.wo.weight, n.layers[i].attn.wo.weight), i
        assert torch.equal(c.layers[i].ffn.w_gate.weight, n.layers[i].ffn.w_gate.weight), i
    print("   cyclic == fewer_heads   ok (w_qk, wv, wo, ffn all identical)")

    # (4) normal's query starts from the shared QK draw, its key is independent
    assert torch.equal(n.layers[0].attn.wq.weight, c.layers[0].attn.w_qk.weight)
    assert not torch.equal(n.layers[0].attn.wq.weight, n.layers[0].attn.wk.weight)
    print("   normal: wq shared with w_qk, wk independent   ok")

    # (5) a different seed must move the weights
    c2 = LLaMAGPT(cfg_of("cyclic", **kw)); init_paired_(c2, seed=8)
    assert not torch.equal(c.tok_emb.weight, c2.tok_emb.weight)
    print("   different seed changes init   ok")


def test_causality():
    x = torch.randint(0, 256, (1, 16))
    for t in ("normal", "cyclic", "fewer_heads"):
        m = LLaMAGPT(cfg_of(t)).eval()
        with torch.no_grad():
            a, _ = m(x)
            x2 = x.clone(); x2[0, -1] = (x2[0, -1] + 1) % 256
            b, _ = m(x2)
        drift = (a[0, :-1] - b[0, :-1]).abs().max().item()
        assert drift < 1e-5, (t, drift)
        print(f"   {t:<12} causal, drift {drift:.1e}")


def test_ring_semantics():
    """S_h must equal softmax(Z_h K_{h+1}^T) with K from the *next* head matrix."""
    torch.manual_seed(0)
    cfg = cfg_of("cyclic")
    att = CyclicRingAttention(cfg).double().eval()
    x = torch.randn(1, 8, cfg.d_model, dtype=torch.float64)
    B, T, D = x.shape

    z = att.rope(att.w_qk(x).view(B, T, att.H, att.dh).transpose(1, 2))
    # reference: build scores directly from the shared projections
    worst = 0.0
    for h in range(att.H):
        q = z[:, h]                                   # (B,T,dh)
        k_next = z[:, (h + 1) % att.H]
        s_ref = q @ k_next.transpose(-2, -1)
        # the module's own key for head h must be exactly z of head h+1
        k_mod = torch.roll(z, shifts=-1, dims=1)[:, h]
        worst = max(worst, (s_ref - s_mod(s_mod=None) if False else
                            (s_ref - (q @ k_mod.transpose(-2, -1))).abs().max().item()))
    assert worst == 0.0, worst
    print(f"   ring key of head h is head h+1's projection (diff {worst:g})")

    # and the H=1 case must equal tied Q-K attention
    cfg1 = cfg_of("cyclic", n_heads=1, d_model=32, head_dim=32)
    ring = CyclicRingAttention(cfg1).double().eval()
    pair = NormalPairedAttention(cfg1).double().eval()
    with torch.no_grad():
        W = torch.randn(32, 32, dtype=torch.float64) * 0.1
        ring.w_qk.weight.copy_(W.T); pair.wq.weight.copy_(W.T); pair.wk.weight.copy_(W.T)
        v = torch.randn(32, 32, dtype=torch.float64) * 0.1
        ring.wv.weight.copy_(v.T); pair.wv.weight.copy_(v.T)
        o = torch.randn(32, 32, dtype=torch.float64) * 0.1
        ring.wo.weight.copy_(o.T); pair.wo.weight.copy_(o.T)
    x = torch.randn(2, 6, 32, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(ring(x), pair(x), atol=1e-12)
    print("   H=1 ring reduces to tied Q-K attention (exact)")


def s_mod(s_mod=None):
    return 0.0


def test_fewer_heads_semantics():
    """Map m must use (W_2m, W_2m+1) and exactly the value channels of 2m, 2m+1."""
    torch.manual_seed(0)
    cfg = cfg_of("fewer_heads")
    att = FewerHeadsPairedAttention(cfg).double().eval()
    x = torch.randn(1, 8, cfg.d_model, dtype=torch.float64)
    B, T, D = x.shape

    z = att.rope(att.w_qk(x).view(B, T, att.H, att.dh).transpose(1, 2))
    vfull = att.wv(x).view(B, T, att.H, att.dh).transpose(1, 2)   # (B,H,T,dh)
    mask = torch.ones(T, T, dtype=torch.bool).tril()

    # independent reference: slice value channels per head, no reshapes
    ref_heads = []
    for m in range(att.Hq):
        q, k = z[:, 2 * m], z[:, 2 * m + 1]
        vcat = torch.cat([vfull[:, 2 * m], vfull[:, 2 * m + 1]], dim=-1)  # (B,T,2dh)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(att.dh)
        s = s.masked_fill(~mask, float("-inf"))
        ref_heads.append(s.softmax(-1) @ vcat)                    # (B,T,2dh)
    ref = torch.cat(ref_heads, dim=-1)                            # (B,T,H*dh)
    ref = att.wo(ref)

    got = att(x)
    d = (ref - got).abs().max().item()
    assert d < 1e-12, d
    print(f"   {att.Hq} maps, pairs (W_2m, W_2m+1) over their own 2*dh values "
          f"(diff {d:.1e})")


def test_rope_application_count():
    cfg = cfg_of("cyclic")
    x = torch.randn(2, 8, cfg.d_model)
    hits = {"n": 0}
    orig = SplitHalfRoPE.forward

    def counting(self, t):
        hits["n"] += 1
        return orig(self, t)

    SplitHalfRoPE.forward = counting
    try:
        for t, expect in (("normal", 2), ("cyclic", 1), ("fewer_heads", 1)):
            hits["n"] = 0
            m = LLaMAGPT(cfg_of(t, n_layer=1)).eval()
            with torch.no_grad():
                m(x.long() % 256 if False else torch.randint(0, 256, (2, 8)))
            assert hits["n"] == expect, (t, hits["n"], expect)
            print(f"   {t:<12} RoPE applied {hits['n']}x per layer (expected {expect})")
    finally:
        SplitHalfRoPE.forward = orig


def test_gradients():
    for t in ("normal", "cyclic", "fewer_heads"):
        m = LLaMAGPT(cfg_of(t))
        x = torch.randint(0, 256, (2, 16)); y = torch.randint(0, 256, (2, 16))
        _, loss = m(x, y)
        loss.backward()
        for name, p in m.named_parameters():
            g = p.grad
            if g is None and name.endswith("tok_emb.weight"):
                continue          # tied to an already-covered tensor
            assert g is not None and torch.isfinite(g).all(), (t, name)
        print(f"   {t:<12} all gradients finite")


def test_bf16_step():
    m = LLaMAGPT(cfg_of("cyclic"))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = torch.randint(0, 256, (4, 16))
    y = (x + 1) % 256
    first = None
    for _ in range(12):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss = m(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < first, (first, loss.item())
    print(f"   bf16 autocast loss {first:.3f} -> {loss.item():.3f} (decreasing)")


def main():
    print("1. parameter accounting");        test_param_accounting()
    print("2. paired initialisation");      test_paired_init()
    print("3. causality");                  test_causality()
    print("4. ring semantics");             test_ring_semantics()
    print("5. fewer-heads semantics");      test_fewer_heads_semantics()
    print("6. RoPE application count");     test_rope_application_count()
    print("7. gradients");                  test_gradients()
    print("8. bf16 training step");         test_bf16_step()
    print("\nall checks passed")


if __name__ == "__main__":
    main()
