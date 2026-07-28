#!/bin/bash
set -e

CONFIG=${1:-./configs/unsloth_qwen3.yaml}

echo "Starting Unsloth fine-tuning with config: $CONFIG"
python src/train_unsloth.py --config "$CONFIG"
