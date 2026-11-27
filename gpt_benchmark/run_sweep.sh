#!/bin/bash
# Full sweep: 3 models x 5 seeds, DDP across 2 GPUs, 12000 steps each.
#
#   ./run_sweep.sh
#
# Detached with setsid+nohup so an SSH drop cannot kill it. Resumable:
# run_cyclic_gpt_benchmark.py skips (model, seed) cells already present in
# all_runs_metrics.json, so re-running this script continues where it stopped.
#
# Data: --data_dir takes precedence; otherwise DATA_DIR is honoured, then
# ./data, then the path this benchmark was originally developed against.
set -u
cd "$(dirname "$0")" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p results_cyclic_gpt logs

exec > >(tee -a logs/sweep_master.log) 2>&1
echo "=== sweep start $(date) ==="
torchrun --nproc_per_node=2 --standalone \
    run_cyclic_gpt_benchmark.py \
    --models normal cyclic fewer_heads \
    --seeds 0 1 2 3 4 \
    --n_layer 8 --d_model 384 --n_heads 6 --ffn_hidden 1024 \
    --seq_len 512 --batch_size 64 --micro_batch_size 16 \
    --max_steps 12000 --eval_interval 500 --eval_batches 50 \
    --data_dir "${DATA_DIR:-./data}" \
    --out_dir ./results_cyclic_gpt
rc=$?
echo "=== sweep finished rc=$rc $(date) ==="
