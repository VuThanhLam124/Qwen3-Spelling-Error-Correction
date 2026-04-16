"""
Mục đích file: finetune Qwen3 bằng LoRA/QLoRA trên dữ liệu sửa lỗi chính tả.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

try:
    import wandb
except Exception:
    wandb = None

from src.data_ingest import load_sources, load_yaml, sample_rows
from src.model import attach_lora, load_base_model, load_tokenizer
from src.pipeline import tokenize_sft_example


@dataclass
class CausalLMCollator:
    """
    - arg/input: tokenizer dùng để pad batch.
    - output: callable collator.
    - mục đích của hàm: pad input_ids/attention_mask/labels cho Trainer.
    """
    tokenizer: Any

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """
        - arg/input: một batch features đã token hóa.
        - output: batch tensor đã pad.
        - mục đích của hàm: ghép batch cho causal LM.
        """
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    """
    - arg/input: seed số nguyên.
    - output: không có.
    - mục đích của hàm: cố định ngẫu nhiên để thí nghiệm ổn định hơn.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_wandb(cfg: Dict[str, Any], train_cfg: Dict[str, Any]) -> str:
    """
    - arg/input: config tổng và config train.
    - output: chuỗi report_to cho Trainer.
    - mục đích của hàm: bật hoặc tắt wandb tracking an toàn.
    """
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return "none"

    if wandb is None:
        raise ImportError("Thiếu thư viện wandb. Hãy cài requirements.txt trước.")

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    run_name = wandb_cfg.get("run_name") or os.path.basename(train_cfg["output_dir"])
    wandb.init(
        project=wandb_cfg.get("project", "qwen-spell-correction"),
        entity=wandb_cfg.get("entity"),
        name=run_name,
        config=cfg,
    )
    return "wandb"


def build_hf_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    """
    - arg/input: list record chuẩn hóa.
    - output: Hugging Face Dataset.
    - mục đích của hàm: đổi list python sang Dataset để map tokenization.
    """
    return Dataset.from_list(rows)


def main(config_path: str) -> None:
    """
    - arg/input: đường dẫn config finetune.
    - output: không có.
    - mục đích của hàm: chạy toàn bộ pipeline finetune.
    """
    cfg = load_yaml(config_path)
    seed = cfg["train"].get("seed", cfg["data"].get("seed", 42))
    set_seed(seed)

    tokenizer = load_tokenizer(
        cfg["model"]["base_model"],
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )

    train_rows = load_sources(cfg["data"]["train_sources"], seed=seed)
    train_rows = sample_rows(train_rows, cfg["data"].get("max_train_samples"), seed=seed)

    eval_rows = load_sources(cfg["data"].get("eval_sources", []), seed=seed)
    eval_rows = sample_rows(eval_rows, cfg["data"].get("max_eval_samples"), seed=seed)

    train_ds = build_hf_dataset(train_rows)
    eval_ds = build_hf_dataset(eval_rows) if eval_rows else None

    system_prompt = cfg["prompt"]["system_prompt"]
    max_length = cfg["train"].get("max_length", 192)

    train_ds = train_ds.map(
        lambda x: tokenize_sft_example(x, tokenizer, system_prompt, max_length),
        remove_columns=train_ds.column_names,
        desc="Tokenize train",
    )

    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda x: tokenize_sft_example(x, tokenizer, system_prompt, max_length),
            remove_columns=eval_ds.column_names,
            desc="Tokenize eval",
        )

    model = load_base_model(cfg["model"], training=True)
    model = attach_lora(model, cfg["lora"])
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    if cfg["train"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_cfg = cfg["train"]
    report_to = setup_wandb(cfg, train_cfg)

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        eval_steps=train_cfg.get("eval_steps", 100),
        eval_strategy=train_cfg.get("evaluation_strategy", "steps"),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        report_to=report_to,
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", False),
        remove_unused_columns=False,
    )

    data_collator = CausalLMCollator(tokenizer)
    
    print("train samples:", len(train_ds))
    print("eval samples:", 0 if eval_ds is None else len(eval_ds))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

    summary = {
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "output_dir": train_cfg["output_dir"],
        "base_model": cfg["model"]["base_model"],
    }
    os.makedirs(train_cfg["output_dir"], exist_ok=True)
    with open(os.path.join(train_cfg["output_dir"], "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if cfg.get("wandb", {}).get("enabled", False) and wandb is not None and wandb.run is not None:
        wandb.log(summary)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/finetune.yaml")
    args = parser.parse_args()
    main(args.config)
