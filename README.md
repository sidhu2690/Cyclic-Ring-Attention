# Cyclic-ring attention: reproduction and scale benchmark

Two self-contained parts of the paper *"Your Query Is My Key: Halving Attention
Parameters via Cross-Head Cyclic Rings"*.

```
paper_repro/        the paper's own experiments, CPU, minutes to hours
gpt_benchmark/      a 3-model scale comparison on FineWeb, DDP, 2 GPUs
```

Each folder is independent — its own `requirements.txt`, no shared imports.

## paper_repro

Five files: `cyclic_core.py` (models, trainer, reporting) plus one script per
table.

```bash
python exp3_memory.py                              # 10 s
python exp1_mqar.py --mode grokking --seeds 0 1 2  # 20 min
python exp2_charlm.py --seeds 0 1 2                # 1 h
python exp4_expressiveness.py --mode all --seeds 0 1 2   # 14 h
```

Every script prints `PASS` / `FAIL` / `INFO` per claim and writes JSON to
`results_repro/`. It uses CUDA if present, CPU otherwise.

| script | paper | checks |
|---|---|---|
| `exp1_mqar.py` | Table I, Fig. 3 | MQAR over 8 configurations; the grokking trajectory |
| `exp2_charlm.py` | Table II | char-level LM, 5 configurations, exact parameter parity |
| `exp3_memory.py` | IV-B, VII-C, VIII | Eq. (10) identity in fp64, Table III memory, 7B extrapolation |
| `exp4_expressiveness.py` | VII-D, Table IV | 408 cells: 96 formal + 24 ListOps + 288 length-generalisation |

## gpt_benchmark

`normal` (H heads, independent Q/K) vs `cyclic` (one shared QK projection
routed around a ring) vs `fewer_heads` (H/2 paired heads). The last two spend
exactly the same QK parameters, so the comparison is like for like.

```bash
python run_cyclic_gpt_benchmark.py --dry_run_checks   # correctness, no data needed
./run_sweep.sh                                        # 3 models x 5 seeds, 2 GPUs
```

Needs FineWeb `uint16` shards as `fineweb_train_*.bin` / `fineweb_val_*.bin` in
`DATA_DIR` (default `./data`). Set `DATA_DIR` to override.

## Notes

* **No corpus is shipped.** `exp2_charlm.py` builds a deterministic 146-symbol
  corpus of the paper's size, so parameter counts are exact but its
  cross-entropy is *not* comparable to the paper's.
* **`cycle3` is a binary task** in `exp4` (`y = 1{Σw ≡ 0 mod 3}`), whereas the
  paper's Table IV uses a three-way label where chance is 1/3.
* **CUDA peak memory** is allocator- and host-dependent, so it is reported,
  never asserted.
