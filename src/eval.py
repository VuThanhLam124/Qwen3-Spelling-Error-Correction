"""
Mục đích file: đánh giá model trên bộ test sửa lỗi chính tả.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
import os
from typing import Any, Dict, List, Sequence

from src.data_ingest import load_sources, load_yaml, sample_rows, save_jsonl
from src.model import load_model_for_inference, load_tokenizer
from src.pipeline import predict_text

try:
    import wandb
except Exception:
    wandb = None


def levenshtein(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    """
    - arg/input: hai chuỗi hoặc hai dãy token.
    - output: khoảng cách Levenshtein.
    - mục đích của hàm: tính CER/WER đơn giản không phụ thuộc thư viện ngoài.
    """
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def maybe_init_wandb(cfg: Dict[str, Any], mode: str) -> None:
    """
    - arg/input: config tổng và tên mode.
    - output: không có.
    - mục đích của hàm: mở wandb run tùy chọn cho eval.
    """
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False) or wandb is None:
        return

    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)

    base_run_name = wandb_cfg.get("run_name") or "qwen-spell-correction"
    wandb.init(
        project=wandb_cfg.get("project", "qwen-spell-correction"),
        entity=wandb_cfg.get("entity"),
        name=f"{base_run_name}-{mode}",
        config=cfg,
        reinit=True,
    )


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    - arg/input: list record đã có prediction.
    - output: dict metrics.
    - mục đích của hàm: tổng hợp các chỉ số chính cho bài toán correction.
    """
    if not rows:
        return {}

    exact_matches = 0
    char_error_sum = 0.0
    word_error_sum = 0.0
    no_change_total = 0
    no_change_correct = 0

    per_source = defaultdict(lambda: {"count": 0, "exact_match": 0})

    for row in rows:
        pred = row["prediction"].strip()
        target = row["target_text"].strip()
        source_name = row.get("source_name", "unknown")

        per_source[source_name]["count"] += 1
        if pred == target:
            exact_matches += 1
            per_source[source_name]["exact_match"] += 1

        char_dist = levenshtein(list(pred), list(target))
        word_dist = levenshtein(pred.split(), target.split())
        char_error_sum += char_dist / max(len(target), 1)
        word_error_sum += word_dist / max(len(target.split()), 1)

        if row["input_text"].strip() == target:
            no_change_total += 1
            if pred == target:
                no_change_correct += 1

    metrics = {
        "num_samples": len(rows),
        "exact_match": exact_matches / len(rows),
        "cer": char_error_sum / len(rows),
        "wer": word_error_sum / len(rows),
        "no_change_accuracy": (no_change_correct / no_change_total) if no_change_total else None,
        "per_source": {
            key: {
                "count": value["count"],
                "exact_match": value["exact_match"] / max(value["count"], 1),
            }
            for key, value in per_source.items()
        },
    }
    return metrics


def main(config_path: str) -> None:
    """
    - arg/input: đường dẫn config eval.
    - output: không có.
    - mục đích của hàm: chạy eval và lưu metrics/predictions.
    """
    cfg = load_yaml(config_path)
    eval_cfg = cfg["eval"]
    maybe_init_wandb(cfg, "eval")

    tokenizer = load_tokenizer(cfg["model"]["base_model"], cfg["model"].get("trust_remote_code", True))
    model = load_model_for_inference(cfg["model"])

    rows = load_sources(eval_cfg["sources"])
    rows = sample_rows(rows, eval_cfg.get("max_eval_samples"))

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

    metrics = compute_metrics(outputs)
    save_jsonl(outputs, eval_cfg["save_predictions_path"])

    with open(eval_cfg["save_metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if cfg.get("wandb", {}).get("enabled", False) and wandb is not None and wandb.run is not None:
        flat_metrics = {
            "eval/num_samples": metrics.get("num_samples", 0),
            "eval/exact_match": metrics.get("exact_match", 0.0),
            "eval/cer": metrics.get("cer", 0.0),
            "eval/wer": metrics.get("wer", 0.0),
        }
        if metrics.get("no_change_accuracy") is not None:
            flat_metrics["eval/no_change_accuracy"] = metrics["no_change_accuracy"]
        wandb.log(flat_metrics)
        wandb.finish()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/eval.yaml")
    args = parser.parse_args()
    main(args.config)
