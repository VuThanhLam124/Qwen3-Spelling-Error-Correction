# QWEN3-SPELLING-ERROR-CORRECTION

Khung code tối giản cho bài toán sửa lỗi chính tả tiếng Việt bằng `Qwen/Qwen3-0.6B` + LoRA/QLoRA.

## Cấu trúc

```text
config/
  eval.yaml
  finetune.yaml
  infer.yaml
src/
  __init__.py
  app.py
  data_ingest.py
  eval.py
  finetune.py
  infer.py
  model.py
  pipeline.py
LICENSE
README.md
requirements.txt
```

## Ý tưởng hiện tại

- Finetune generic trước với `coung21/vi-spelling-correction`
- Eval ngoài mẫu trên `nguyenthanhasia/vsec-vietnamese-spell-correction`
- Chưa gắn dữ liệu địa chỉ; khi có thì thêm dưới dạng file `jsonl`

## Schema dữ liệu local khuyên dùng

```json
{"input_text": "thadnh phố cần thơ", "target_text": "thành phố Cần Thơ"}
```

Có thể thêm field phụ như `domain`, `source_type`, `needs_correction`, nhưng không bắt buộc.

## Cài thư viện

```bash
pip install -r requirements.txt
```

## Dùng Weights & Biases (wandb)

Không hardcode key vào code hoặc config. Hãy export qua biến môi trường:

```bash
export WANDB_API_KEY="<your_wandb_api_key>"
```

Sau đó chỉnh `wandb.enabled: true` trong `config/finetune.yaml` hoặc `config/eval.yaml`.

## Finetune

```bash
python -m src.finetune --config config/finetune.yaml
```

Muốn resume từ một checkpoint cụ thể, đặt trong YAML:

```yaml
train:
  output_dir: outputs/qwen3-0.6b-spell-lora
  resume_from_checkpoint: outputs/qwen3-0.6b-spell-lora/checkpoint-1200
```

Nếu `resume_from_checkpoint: null`, code sẽ tự tìm `checkpoint-*` mới nhất trong `train.output_dir`.

Khi resume, cấu hình LoRA thực tế sẽ được đọc từ checkpoint để khớp shape adapter cũ. Vì vậy nếu YAML hiện tại khác `r`, `alpha`, `dropout` hoặc `target_modules`, code sẽ tự ghi đè các giá trị này theo checkpoint.

## Infer một câu

```bash
python -m src.infer --config config/infer.yaml --text "thadnh phố cần thơ"
```

## Infer theo file

1. Điền `io.input_file` trong `config/infer.yaml`
2. Chạy:

```bash
python -m src.infer --config config/infer.yaml
```

## Eval

```bash
python -m src.eval --config config/eval.yaml
```

## App CLI nhỏ

```bash
python -m src.app --config config/infer.yaml
```

## Gợi ý khi thêm dữ liệu địa chỉ sau này

- Chia `train/dev/test` theo **địa chỉ gốc** trước khi sinh nhiễu
- Thêm cả mẫu `no-change`
- Lưu dữ liệu địa chỉ ở `jsonl`
- Gắn thêm nguồn đó vào `data.train_sources` hoặc `eval.sources`
