# LLM Fine-tuning: Qwen3 / Llama 3 with LoRA & QLoRA
## Why This Project
This project showcases your ability to train, fine-tune, and deploy large language models from scratch—the core differentiator for an AI Engineer.
这个项目展示你能够从零开始训练、调优和部署大语言模型——这正是 AI Engineer 的核心区分度。

## Tech Stack

- **Models**: Qwen3, Llama 3
- **PEFT**: LoRA, QLoRA, rsLoRA (Unsloth)
- **Frameworks**: Hugging Face Transformers, PEFT, TRL, Unsloth
- **Quantization**: BitsAndBytes 4-bit (NF4 + double quant)
- **Tracking**: Weights & Biases, TensorBoard
- **Deployment**: merge adapter → vLLM / Ollama

## Project Structure

```
llm-finetuning-qwen3/
├── configs/
│   ├── lora_qwen3.yaml          # Standard LoRA / QLoRA config
│   └── unsloth_qwen3.yaml       # Unsloth accelerated config
├── data/
│   ├── alpaca_zh.jsonl          # Example Chinese instruction dataset
│   └── .gitkeep
├── notebooks/
│   └── 01_lora_qlora_tutorial.ipynb  # Step-by-step notebook
├── scripts/
│   ├── download_model.py        # Cache base model locally
│   ├── run_lora.sh              # One-command LoRA training
│   └── run_unsloth.sh           # One-command Unsloth training
├── src/
│   ├── data_utils.py            # Dataset loading & formatting (Alpaca / ShareGPT)
│   ├── train_lora.py            # Standard PEFT + TRL trainer
│   ├── train_unsloth.py         # Unsloth fast trainer
│   ├── merge_adapter.py         # Merge adapter into base model
│   └── inference.py             # Load adapter / merged model and generate
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/WenRui-MA7238/llm-finetuning-qwen3.git
cd llm-finetuning-qwen3
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Windows 安装 `bitsandbytes` 可能遇到困难，可改用 WSL2 或 CUDA 12.1 预编译 wheel。

### 2. Configure

```bash
cp .env.example .env
# Edit HF_TOKEN, WANDB_PROJECT, etc.
```

### 3. Train

**Option A: Standard LoRA / QLoRA**

```bash
bash scripts/run_lora.sh configs/lora_qwen3.yaml
```

**Option B: Unsloth (2-5x faster, recommended for single GPU)**

```bash
bash scripts/run_unsloth.sh configs/unsloth_qwen3.yaml
```

### 4. Merge Adapter for Deployment

```bash
python src/merge_adapter.py \
  --adapter ./outputs/qwen3_lora/final_adapter \
  --output ./outputs/qwen3_lora_merged
```

### 5. Inference

```bash
python src/inference.py \
  --model ./outputs/qwen3_lora_merged \
  --prompt "### Instruction:\n解释 LoRA 微调\n\n### Response:\n"
```

## Hardware Requirements

| Method | 7B Model | 13B Model | Notes |
|--------|----------|-----------|-------|
| LoRA (16-bit) | 24 GB VRAM | 40 GB VRAM | Full bf16 base model |
| QLoRA (4-bit) | 8-10 GB VRAM | 16-20 GB VRAM | Recommended |
| Unsloth QLoRA | 6-8 GB VRAM | 12-16 GB VRAM | Fastest |

## Key Concepts Demonstrated

1. **LoRA**: Low-rank adaptation with trainable rank decomposition matrices.
2. **QLoRA**: 4-bit quantized base model + LoRA + double quantization + paged optimizers.
3. **PEFT**: Parameter-efficient fine-tuning library from Hugging Face.
4. **SFT**: Supervised fine-tuning on instruction-following datasets.
5. **Unsloth**: Hand-optimized kernels for faster LoRA training.
6. **Adapter merging**: Combine LoRA weights back into base model for easier serving.

## Production Notes

- Always evaluate on a held-out set; monitor perplexity and instruction-following accuracy.
- Use `max_grad_norm` and gradient clipping to prevent loss spikes.
- Save checkpoints frequently; QLoRA training can be unstable with high learning rates.
- For multi-GPU, switch `device_map="auto"` to `accelerate launch` with DeepSpeed/FSDP.
- Merge adapter before deploying with vLLM for maximum throughput.

## Evaluation

```bash
# Coming soon: lm-eval harness integration
lm_eval --model hf --model_args pretrained=./outputs/qwen3_lora_merged --tasks hellaswag,arc_easy
```

## License

MIT
