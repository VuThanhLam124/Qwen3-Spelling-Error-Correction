"""
Mục đích file: suy luận một câu hoặc một file dữ liệu với model đã finetune.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.data_ingest import save_jsonl, load_yaml
from src.model import load_model_for_inference, load_tokenizer
from src.pipeline import predict_text


def load_infer_items(path: str | Path) -> List[Dict[str, Any]]:
    """
    - arg/input: đường dẫn file txt/jsonl.
    - output: list item có trường input_text.
    - mục đích của hàm: đọc dữ liệu đầu vào cho bước infer hàng loạt.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item.get("input_text") or item.get("text") or item.get("source") or ""
                if text:
                    item["input_text"] = text
                    rows.append(item)
        return rows

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append({"input_text": text})
    return rows


def run_single_text(cfg: Dict[str, Any], text: str) -> None:
    """
    - arg/input: config infer và một câu text.
    - output: in prediction ra màn hình.
    - mục đích của hàm: suy luận nhanh cho một câu.
    """
    tokenizer = load_tokenizer(cfg["model"]["base_model"], cfg["model"].get("trust_remote_code", True))
    model = load_model_for_inference(cfg["model"])
    pred = predict_text(
        model,
        tokenizer,
        text,
        cfg["prompt"]["system_prompt"],
        cfg["generation"],
        enable_thinking=cfg["prompt"].get("enable_thinking", False),
    )
    print(pred)


def run_file(cfg: Dict[str, Any], input_file: str, output_file: str) -> None:
    """
    - arg/input: config infer, file input và file output.
    - output: ghi jsonl predictions.
    - mục đích của hàm: suy luận hàng loạt cho một file dữ liệu.
    """
    tokenizer = load_tokenizer(cfg["model"]["base_model"], cfg["model"].get("trust_remote_code", True))
    model = load_model_for_inference(cfg["model"])

    rows = load_infer_items(input_file)
    outputs = []
    for row in rows:
        prediction = predict_text(
            model,
            tokenizer,
            row["input_text"],
            cfg["prompt"]["system_prompt"],
            cfg["generation"],
            enable_thinking=cfg["prompt"].get("enable_thinking", False),
        )
        outputs.append({**row, "prediction": prediction})

    save_jsonl(outputs, output_file)
    print(f"Đã lưu dự đoán: {output_file}")


def main(config_path: str, text: str | None = None, adapter_path: str | None = None) -> None:
    """
    - arg/input: đường dẫn config, text tùy chọn, và adapter_path override tùy chọn.
    - output: không có.
    - mục đích của hàm: chọn mode suy luận một câu hoặc cả file.
    """
    cfg = load_yaml(config_path)

    # Override adapter_path nếu truyền qua CLI (vd: test tại checkpoint cụ thể).
    if adapter_path is not None:
        cfg["model"]["adapter_path"] = adapter_path

    if text:
        run_single_text(cfg, text)
        return

    input_file = cfg["io"].get("input_file")
    output_file = cfg["io"].get("output_file", "outputs/infer_predictions.jsonl")
    if not input_file:
        raise ValueError("Hãy truyền --text hoặc điền io.input_file trong config/infer.yaml")
    run_file(cfg, input_file, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/infer.yaml")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        dest="adapter_path",
        help="Override adapter_path trong config (vd: outputs/.../checkpoint-6000).",
    )
    args = parser.parse_args()
    main(args.config, args.text, args.adapter_path)
