#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLUE_DIR="${GLUE_DIR:-$REPO_ROOT/glue_data}"
TASK_NAME="${TASK_NAME:-RTE}"
OUTPUT_DIR="${1:-$REPO_ROOT/task1/results}"

mkdir -p "$OUTPUT_DIR"

python3 "$REPO_ROOT/task1/run_glue.py" \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name "$TASK_NAME" \
  --do_train \
  --do_eval \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir
