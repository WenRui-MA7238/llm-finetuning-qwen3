"""
Standard LoRA / QLoRA fine-tuning script for Qwen3.
Uses PEFT + Transformers + TRL.
"""

import os
import argparse
from pathlib import Path

import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from data_utils import build_sft_dataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    model_name = cfg["base_model"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config for QLoRA
    bnb_config = None
    if cfg.get("qlora", {}).get("enabled", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg["qlora"]["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(
                torch, cfg["qlora"]["bnb_4bit_compute_dtype"]
            ),
            bnb_4bit_use_double_quant=cfg["qlora"]["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=cfg["qlora"]["bnb_4bit_quant_type"],
        )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias=cfg["lora"].get("bias", "none"),
        task_type=cfg["lora"].get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset
    train_dataset = build_sft_dataset(
        dataset_path=cfg["dataset_path"],
        template=cfg.get("template", "alpaca"),
        tokenizer=tokenizer,
        max_seq_length=cfg["max_seq_length"],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=cfg.get("weight_decay", 0.0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 3),
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", False),
        report_to=cfg.get("report_to", "none"),
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        max_seq_length=cfg["max_seq_length"],
        dataset_text_field="text",
    )

    trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(output_dir / "final_adapter")
    tokenizer.save_pretrained(output_dir / "final_adapter")
    print(f"Training complete. Adapter saved to {output_dir / 'final_adapter'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/lora_qwen3.yaml",
        help="Path to YAML training config",
    )
    args = parser.parse_args()
    main(args.config)
