#!/bin/bash
set -e

CONFIG=${1:-./configs/lora_qwen3.yaml}

echo "Starting LoRA fine-tuning with config: $CONFIG"
python src/train_lora.py --config "$CONFIG"
