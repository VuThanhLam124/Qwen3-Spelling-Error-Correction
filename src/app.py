"""
Mục đích file: giao diện CLI nhỏ để thử model sửa lỗi bằng tay.
"""

from __future__ import annotations

import argparse

from src.data_ingest import load_yaml
from src.model import load_model_for_inference, load_tokenizer
from src.pipeline import predict_text


def main(config_path: str) -> None:
    """
    - arg/input: đường dẫn config infer.
    - output: không có.
    - mục đích của hàm: mở vòng lặp nhập câu và xem kết quả ngay.
    """
    cfg = load_yaml(config_path)
    tokenizer = load_tokenizer(cfg["model"]["base_model"], cfg["model"].get("trust_remote_code", True))
    model = load_model_for_inference(cfg["model"])

    print("Nhập câu cần sửa. Gõ 'exit' để thoát.")
    while True:
        text = input("\nInput: ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        prediction = predict_text(model, tokenizer, text, cfg["prompt"]["system_prompt"], cfg["generation"])
        print(f"Output: {prediction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/infer.yaml")
    args = parser.parse_args()
    main(args.config)
