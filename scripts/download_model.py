"""
Download a base model locally to avoid repeated downloads during experiments.
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_name: str, cache_dir: str = None):
    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    print("Download complete.")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()
    main(args.model, args.cache_dir)
