"""
Mục đích file: đọc dữ liệu từ Hugging Face hoặc file local,
chuẩn hóa về cùng schema input_text/target_text để train và eval.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from datasets import load_dataset


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    - arg/input: đường dẫn file yaml.
    - output: dict cấu hình.
    - mục đích của hàm: đọc file cấu hình yaml.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: Any) -> str:
    """
    - arg/input: text bất kỳ.
    - output: chuỗi đã chuẩn hóa khoảng trắng cơ bản.
    - mục đích của hàm: làm sạch nhẹ trước khi train/eval.
    """
    if text is None:
        return ""
    text = str(text).replace(" ", " ").replace("	", " ")
    return " ".join(text.split()).strip()


def _guess_pair_keys(example: Dict[str, Any]) -> Tuple[str, str]:
    """
    - arg/input: một record dữ liệu thô.
    - output: cặp tên cột input/target.
    - mục đích của hàm: tự đoán cột khi nguồn dữ liệu không có rule riêng.
    """
    candidates = [
        ("input_text", "target_text"),
        ("source", "target"),
        ("text", "corrected_text"),
        ("original", "normalized"),
        ("input", "output"),
        ("src", "tgt"),
    ]
    keys = set(example.keys())
    for input_key, target_key in candidates:
        if input_key in keys and target_key in keys:
            return input_key, target_key
    raise ValueError(f"Không đoán được cột input/target từ keys: {sorted(keys)}")


def _convert_example(example: Dict[str, Any], source_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    - arg/input: record thô và config của nguồn dữ liệu.
    - output: record chuẩn hóa.
    - mục đích của hàm: đưa mọi nguồn về cùng schema dùng chung.
    """
    source_name = source_cfg.get("name") or source_cfg.get("path") or "unknown_source"

    if source_name == "coung21/vi-spelling-correction":
        input_key, target_key = "source", "target"
    elif source_name == "nguyenthanhasia/vsec-vietnamese-spell-correction":
        input_key, target_key = "text", "corrected_text"
    else:
        input_key = source_cfg.get("input_key")
        target_key = source_cfg.get("target_key")
        if not input_key or not target_key:
            input_key, target_key = _guess_pair_keys(example)

    input_text = normalize_text(example.get(input_key))
    target_text = normalize_text(example.get(target_key))

    if not input_text or not target_text:
        raise ValueError(f"Record rỗng ở nguồn {source_name}")

    return {
        "input_text": input_text,
        "target_text": target_text,
        "domain": source_cfg.get("domain", "generic"),
        "source_type": source_cfg.get("source_type", "unknown"),
        "source_name": source_name,
        "needs_correction": input_text != target_text,
    }


def _load_hf_source(source_cfg: Dict[str, Any], seed: int = 42) -> List[Dict[str, Any]]:
    """
    - arg/input: config một nguồn Hugging Face và seed.
    - output: list record chuẩn hóa.
    - mục đích của hàm: tải dataset từ Hugging Face rồi chuẩn hóa record.
    """
    dataset = load_dataset(
        path=source_cfg["name"],
        name=source_cfg.get("subset"),
        split=source_cfg.get("split", "train"),
    )

    if source_cfg.get("shuffle", False):
        dataset = dataset.shuffle(seed=seed)

    max_samples = source_cfg.get("max_samples")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    rows: List[Dict[str, Any]] = []
    for example in dataset:
        try:
            rows.append(_convert_example(example, source_cfg))
        except Exception:
            continue
    return rows


def _load_jsonl_source(source_cfg: Dict[str, Any], seed: int = 42) -> List[Dict[str, Any]]:
    """
    - arg/input: config một nguồn json/jsonl và seed.
    - output: list record chuẩn hóa.
    - mục đích của hàm: tải file local để dùng cho train/eval/infer.
    """
    dataset = load_dataset(
        path="json",
        data_files=source_cfg["path"],
        split="train",
    )

    if source_cfg.get("shuffle", False):
        dataset = dataset.shuffle(seed=seed)

    max_samples = source_cfg.get("max_samples")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    rows: List[Dict[str, Any]] = []
    for example in dataset:
        try:
            rows.append(_convert_example(example, source_cfg))
        except Exception:
            continue
    return rows


def load_sources(sources: Iterable[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """
    - arg/input: danh sách config nguồn dữ liệu và seed.
    - output: list record chuẩn hóa đã merge.
    - mục đích của hàm: tải nhiều nguồn dữ liệu vào một list chung.
    """
    rows: List[Dict[str, Any]] = []
    for source_cfg in sources:
        source_type = source_cfg.get("type", "huggingface")
        if source_type == "huggingface":
            rows.extend(_load_hf_source(source_cfg, seed=seed))
        elif source_type in {"json", "jsonl"}:
            rows.extend(_load_jsonl_source(source_cfg, seed=seed))
        else:
            raise ValueError(f"Chưa hỗ trợ source type: {source_type}")
    return rows


def sample_rows(rows: List[Dict[str, Any]], max_samples: Optional[int], seed: int = 42) -> List[Dict[str, Any]]:
    """
    - arg/input: list record, số lượng tối đa, seed.
    - output: list record đã cắt mẫu.
    - mục đích của hàm: giảm kích thước dữ liệu khi thử nghiệm nhanh.
    """
    if not max_samples or max_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    picked = indices[:max_samples]
    return [rows[i] for i in picked]


def save_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """
    - arg/input: list record và đường dẫn file output.
    - output: không có.
    - mục đích của hàm: lưu dự đoán hoặc dữ liệu chuẩn hóa ra jsonl.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
