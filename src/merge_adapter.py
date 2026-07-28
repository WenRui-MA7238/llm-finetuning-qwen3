"""
Merge a trained LoRA adapter back into the base model for easier deployment.
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(adapter_path: str, output_path: str):
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Base model is stored in adapter config
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()

    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", required=True, help="Path to save merged model")
    args = parser.parse_args()
    merge(args.adapter, args.output)
