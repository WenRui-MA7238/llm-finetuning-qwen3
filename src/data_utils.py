"""
Dataset preprocessing utilities for supervised fine-tuning (SFT).
Supports Alpaca and ShareGPT formats.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset


def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of records."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def format_alpaca(record: Dict, system: str = "") -> str:
    """Format a single record in Alpaca instruction format."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    output = record.get("output", "")

    if input_text:
        prompt = f"{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"{instruction}\n\n### Response:\n"

    if system:
        prompt = f"{system}\n\n{prompt}"

    return prompt + output


def format_sharegpt(record: Dict) -> str:
    """Format a ShareGPT-style conversation into a single string."""
    messages = record.get("messages", record.get("conversations", []))
    formatted = ""
    for msg in messages:
        role = msg.get("from", msg.get("role", ""))
        content = msg.get("value", msg.get("content", ""))
        if role in ("human", "user"):
            formatted += f"### Human:\n{content}\n\n"
        elif role in ("gpt", "assistant"):
            formatted += f"### Assistant:\n{content}\n\n"
    return formatted.strip()


def build_sft_dataset(
    dataset_path: str,
    template: str = "alpaca",
    tokenizer=None,
    max_seq_length: int = 2048,
    system_message: str = "You are a helpful assistant.",
) -> Dataset:
    """
    Build a tokenized SFT dataset.

    Args:
        dataset_path: Path to JSONL file or HuggingFace dataset name.
        template: 'alpaca' or 'sharegpt'.
        tokenizer: HuggingFace tokenizer.
        max_seq_length: Max sequence length.
        system_message: System prompt prepended to Alpaca examples.
    """
    path = Path(dataset_path)
    if path.exists():
        records = load_jsonl(str(path))
        dataset = Dataset.from_list(records)
    else:
        dataset = load_dataset(dataset_path, split="train")

    if template == "alpaca":
        texts = [
            format_alpaca(record, system=system_message) for record in dataset
        ]
    elif template == "sharegpt":
        texts = [format_sharegpt(record) for record in dataset]
    else:
        raise ValueError(f"Unknown template: {template}")

    text_dataset = Dataset.from_dict({"text": texts})

    if tokenizer is None:
        return text_dataset

    def tokenize(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    return text_dataset.map(
        tokenize,
        batched=True,
        remove_columns=text_dataset.column_names,
    )


def preview_dataset(path: str, template: str = "alpaca", n: int = 3) -> None:
    """Print n formatted examples from the dataset."""
    records = load_jsonl(path)
    formatter = format_alpaca if template == "alpaca" else format_sharegpt
    for i, record in enumerate(records[:n], 1):
        print(f"--- Example {i} ---")
        print(formatter(record))
        print()


if __name__ == "__main__":
    preview_dataset("./data/alpaca_zh.jsonl", template="alpaca")
