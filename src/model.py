"""
Mục đích file: nạp tokenizer, model và adapter LoRA cho Qwen3.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _resolve_dtype(dtype_name: str | None) -> torch.dtype:
    """
    - arg/input: tên dtype dạng chuỗi.
    - output: torch.dtype.
    - mục đích của hàm: đổi config yaml sang torch dtype.
    """
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float16


def build_quant_config(model_cfg: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    """
    - arg/input: config phần model.
    - output: BitsAndBytesConfig hoặc None.
    - mục đích của hàm: tạo cấu hình quantization khi chạy QLoRA.
    """
    if not model_cfg.get("load_in_4bit", False):
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=_resolve_dtype(model_cfg.get("bnb_4bit_compute_dtype", "float16")),
    )


def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    """
    - arg/input: tên model và cờ trust_remote_code.
    - output: tokenizer.
    - mục đích của hàm: nạp tokenizer và thiết lập pad token an toàn.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(model_cfg: Dict[str, Any], training: bool = False):
    """
    - arg/input: config model và cờ training.
    - output: model causal LM.
    - mục đích của hàm: nạp model base cho train hoặc infer.
    """
    quant_config = build_quant_config(model_cfg)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        quantization_config=quant_config,
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=None if quant_config else _resolve_dtype(model_cfg.get("bnb_4bit_compute_dtype", "float16")),
    )

    if training and model_cfg.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    return model


def attach_lora(model, lora_cfg: Dict[str, Any]):
    """
    - arg/input: model base và config LoRA.
    - output: model đã gắn LoRA.
    - mục đích của hàm: thêm adapter trainable cho finetune nhẹ.
    """
    peft_cfg = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules"),
    )
    model = get_peft_model(model, peft_cfg)
    return model


def _load_lora_cfg_from_checkpoint_dir(resume_checkpoint: str) -> Dict[str, Any]:
    """
    - arg/input: đường dẫn local tới thư mục checkpoint.
    - output: config LoRA đọc được từ checkpoint.
    - mục đích của hàm: lấy shape/config adapter từ file local, không phụ thuộc HF Hub.
    """
    checkpoint_dir = os.path.abspath(os.path.expanduser(resume_checkpoint))
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    adapter_weights_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")

    if os.path.isfile(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        peft_type = str(raw_cfg.get("peft_type", "")).upper()
        if peft_type and peft_type != "LORA":
            raise ValueError(f"Checkpoint {resume_checkpoint} không phải LoRA adapter.")

        target_modules = raw_cfg.get("target_modules")
        if isinstance(target_modules, str):
            target_modules = [target_modules]

        return {
            "r": raw_cfg.get("r"),
            "alpha": raw_cfg.get("lora_alpha"),
            "dropout": raw_cfg.get("lora_dropout"),
            "target_modules": target_modules,
            "bias": raw_cfg.get("bias", "none"),
        }

    if os.path.isfile(adapter_weights_path):
        from safetensors.torch import load_file

        state_dict = load_file(adapter_weights_path)
        lora_a_keys = [key for key in state_dict.keys() if ".lora_A." in key]
        if not lora_a_keys:
            raise ValueError(f"Không suy ra được rank LoRA từ checkpoint: {resume_checkpoint}")

        rank = int(state_dict[lora_a_keys[0]].shape[0])
        target_modules = sorted(
            {
                match.group(1)
                for key in state_dict.keys()
                for match in [re.search(r"\.([^.]+)\.lora_[AB](?:\.[^.]+)?\.weight$", key)]
                if match is not None
            }
        )

        print(
            "[Resume] Không thấy adapter_config.json; suy ra cấu hình LoRA từ adapter_model.safetensors. "
            "Hãy giữ alpha/dropout trong YAML giống checkpoint gốc."
        )
        return {
            "r": rank,
            "target_modules": target_modules,
            "bias": "none",
        }

    raise FileNotFoundError(
        f"Checkpoint {resume_checkpoint} không có adapter_config.json hoặc adapter_model.safetensors."
    )


def resolve_lora_cfg_for_resume(lora_cfg: Dict[str, Any], resume_checkpoint: str | None) -> Dict[str, Any]:
    """
    - arg/input: config LoRA từ yaml và đường dẫn checkpoint resume.
    - output: config LoRA hiệu lực để dựng model trước khi resume.
    - mục đích của hàm: đồng bộ shape LoRA với checkpoint cũ để tránh lỗi state_dict mismatch.
    """
    effective_cfg = dict(lora_cfg)
    if not resume_checkpoint:
        return effective_cfg

    checkpoint_cfg = _load_lora_cfg_from_checkpoint_dir(resume_checkpoint)

    mismatched_keys = []
    for key, checkpoint_value in checkpoint_cfg.items():
        if checkpoint_value is None:
            continue
        yaml_value = effective_cfg.get(key)
        if yaml_value is not None and yaml_value != checkpoint_value:
            mismatched_keys.append((key, yaml_value, checkpoint_value))
        effective_cfg[key] = checkpoint_value

    if mismatched_keys:
        print("[Resume] Ghi đè cấu hình LoRA từ checkpoint để khớp shape:")
        for key, yaml_value, checkpoint_value in mismatched_keys:
            print(f"  - {key}: yaml={yaml_value} -> checkpoint={checkpoint_value}")

    return effective_cfg


def load_model_for_inference(model_cfg: Dict[str, Any]):
    """
    - arg/input: config model.
    - output: model đã gắn adapter nếu có.
    - mục đích của hàm: nạp model dùng cho infer và eval.
    """
    model = load_base_model(model_cfg, training=False)
    adapter_path = model_cfg.get("adapter_path")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model
