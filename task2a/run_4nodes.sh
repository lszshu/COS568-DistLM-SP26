#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <rank:0-3> <master_ip> <master_port> [output_dir]"
  exit 1
fi

RANK="$1"
MASTER_IP="$2"
MASTER_PORT="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLUE_DIR="${GLUE_DIR:-$REPO_ROOT/glue_data}"
TASK_NAME="${TASK_NAME:-RTE}"
OUTPUT_DIR="${4:-$REPO_ROOT/task2a/results}"

mkdir -p "$OUTPUT_DIR"

python3 "$REPO_ROOT/run_glue.py" \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --task_name "$TASK_NAME" \
  --do_train \
  --do_eval \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir \
  --master_ip "$MASTER_IP" \
  --master_port "$MASTER_PORT" \
  --world_size 4 \
  --local_rank "$RANK"
