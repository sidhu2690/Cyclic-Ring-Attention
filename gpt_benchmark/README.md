# gpt_benchmark

FineWeb-scale comparison of `normal`, `cyclic` and `fewer_heads` at
L=8, D=384, H=6, T=512, 12000 steps, 5 seeds, 2 GPUs.

```bash
python run_cyclic_gpt_benchmark.py --dry_run_checks   # no data needed
./run_sweep.sh
```

Data: uint16 token shards named `fineweb_train_*.bin` and `fineweb_val_*.bin`
under `DATA_DIR` (default `./data`). Vocabulary size is detected by an
exhaustive scan of the shards, not a sample.
