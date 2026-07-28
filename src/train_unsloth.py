"""
Accelerated LoRA fine-tuning using Unsloth.
Unsloth is typically 2-5x faster and uses ~80% less memory than standard TRL.
"""

import argparse
from pathlib import Path

import yaml
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

from data_utils import build_sft_dataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    model_name = cfg["base_model"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    max_seq_length = cfg["max_seq_length"]

    # Load model & tokenizer with Unsloth fast patching
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=cfg["lora"].get("use_rslora", False),
    )

    # Dataset
    train_dataset = build_sft_dataset(
        dataset_path=cfg["dataset_path"],
        template=cfg.get("template", "alpaca"),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        weight_decay=cfg.get("weight_decay", 0.01),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 2),
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to=cfg.get("report_to", "none"),
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    trainer.train()

    # Save adapter
    model.save_pretrained(output_dir / "final_adapter")
    tokenizer.save_pretrained(output_dir / "final_adapter")
    print(f"Unsloth training complete. Adapter saved to {output_dir / 'final_adapter'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/unsloth_qwen3.yaml",
        help="Path to YAML training config",
    )
    args = parser.parse_args()
    main(args.config)
