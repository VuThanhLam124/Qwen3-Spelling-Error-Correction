"""
Mục đích file: nạp tokenizer, model và adapter LoRA cho Qwen3.
"""

from __future__ import annotations

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
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg.get("target_modules"),
    )
    model = get_peft_model(model, peft_cfg)
    return model


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
