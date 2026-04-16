"""
Mục đích file: tạo prompt, tokenize cho SFT và sinh dự đoán khi suy luận.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def build_messages(system_prompt: str, input_text: str, target_text: Optional[str] = None) -> List[Dict[str, str]]:
    """
    - arg/input: system prompt, input text và target text tùy chọn.
    - output: list messages theo format chat.
    - mục đích của hàm: dựng hội thoại chuẩn cho Qwen.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text},
    ]
    if target_text is not None:
        messages.append({"role": "assistant", "content": target_text})
    return messages


def _render_prompt(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    """
    - arg/input: tokenizer, list messages, cờ generation prompt.
    - output: prompt string.
    - mục đích của hàm: render chat template của model nếu có.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    prompt = ""
    for msg in messages:
        prompt += f"[{msg['role'].upper()}]\n{msg['content']}\n"
    if add_generation_prompt:
        prompt += "[ASSISTANT]\n"
    return prompt


def tokenize_sft_example(example: Dict[str, Any], tokenizer, system_prompt: str, max_length: int) -> Dict[str, Any]:
    """
    - arg/input: một record, tokenizer, system prompt và max length.
    - output: record đã token hóa với labels masked.
    - mục đích của hàm: chuẩn bị dữ liệu cho causal LM SFT.
    """
    prompt_messages = build_messages(system_prompt, example["input_text"])
    full_messages = build_messages(system_prompt, example["input_text"], example["target_text"])

    prompt_text = _render_prompt(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = _render_prompt(tokenizer, full_messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
        full_ids = full_ids + [eos_id]

    full_ids = full_ids[:max_length]
    attention_mask = [1] * len(full_ids)

    labels = full_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _get_model_device(model) -> torch.device:
    """
    - arg/input: model.
    - output: torch.device.
    - mục đích của hàm: tìm device để đẩy tensor khi infer.
    """
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


@torch.no_grad()
def predict_text(model, tokenizer, input_text: str, system_prompt: str, generation_cfg: Dict[str, Any]) -> str:
    """
    - arg/input: model, tokenizer, input text, system prompt và config generate.
    - output: chuỗi dự đoán đã decode.
    - mục đích của hàm: suy luận một câu sửa lỗi.
    """
    messages = build_messages(system_prompt, input_text)
    prompt_text = _render_prompt(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = _get_model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": generation_cfg.get("max_new_tokens", 96),
        "do_sample": generation_cfg.get("do_sample", False),
        "temperature": generation_cfg.get("temperature", 0.0),
        "top_p": generation_cfg.get("top_p", 1.0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return prediction.strip()
