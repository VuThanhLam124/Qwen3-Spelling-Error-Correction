# Mục đích file: đọc file txt OCR label, parse thành record có cấu trúc,
# và sinh sample noisy-clean cho bài toán sửa lỗi địa chỉ/OCR.

from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_text(text: str) -> str:
    """
    - arg/input: text thô.
    - output: text đã chuẩn hóa khoảng trắng và unicode.
    - mục đích của hàm: làm sạch text trước khi parse và sinh dataset.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def strip_diacritics(text: str) -> str:
    """
    - arg/input: text tiếng Việt có dấu.
    - output: text đã bỏ dấu.
    - mục đích của hàm: tạo noise mất dấu cho dữ liệu OCR/address.
    """
    if not text:
        return text

    # giữ riêng đ/Đ
    text = text.replace("đ", "d").replace("Đ", "D")

    normalized = unicodedata.normalize("NFD", text)
    no_diacritic = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", no_diacritic)


def is_word_token(token: str) -> bool:
    """
    - arg/input: 1 token string.
    - output: True/False.
    - mục đích của hàm: lọc token phù hợp để áp dụng noise mất dấu.
    """
    if not token:
        return False
    if re.fullmatch(r"\d+", token):
        return False
    if re.fullmatch(r"[^\w]+", token):
        return False
    if any(ch.isalpha() for ch in token):
        return True
    return False


def random_remove_diacritics(
    text: str,
    rng: random.Random,
    min_tokens: int = 1,
    max_tokens: int = 3,
    keep_ratio: float = 0.6,
) -> str:
    """
    - arg/input: text sạch, random generator và các tham số noise.
    - output: text đã bị bỏ dấu ngẫu nhiên một vài từ.
    - mục đích của hàm: sinh noise mất dấu cục bộ cho bài toán sửa lỗi địa chỉ.
    """
    if not text:
        return text

    # một phần sample giữ nguyên
    if rng.random() < keep_ratio:
        return text

    tokens = re.findall(r"\w+|[^\w\s]|\s+", text, flags=re.UNICODE)
    candidate_ids = [i for i, tok in enumerate(tokens) if is_word_token(tok)]

    if not candidate_ids:
        return text

    k = rng.randint(min_tokens, min(max_tokens, len(candidate_ids)))
    selected = set(rng.sample(candidate_ids, k))

    new_tokens = []
    for i, tok in enumerate(tokens):
        if i in selected:
            new_tokens.append(strip_diacritics(tok))
        else:
            new_tokens.append(tok)

    return "".join(new_tokens)


def apply_basic_ocr_noise(text: str, rng: random.Random) -> str:
    """
    - arg/input: text sạch và random generator.
    - output: text bị thêm noise OCR nhẹ.
    - mục đích của hàm: mô phỏng lỗi OCR phổ biến nhưng không phá mạnh semantics.
    """
    if not text:
        return text

    text = random_remove_diacritics(text, rng=rng, min_tokens=1, max_tokens=3, keep_ratio=0.4)

    # bỏ một số dấu phẩy/dấu hai chấm với xác suất nhỏ
    if rng.random() < 0.3:
        text = text.replace(", ", " ", 1)

    if rng.random() < 0.15:
        text = text.replace(": ", ":").replace(":", "")

    # đôi khi viết thường toàn bộ
    if rng.random() < 0.1:
        text = text.lower()

    # đôi khi thêm/xóa khoảng trắng nhẹ
    if rng.random() < 0.15:
        text = re.sub(r"\s+", " ", text)

    return normalize_text(text)


def record_to_full_text(record: Dict[str, Any]) -> str:
    """
    - arg/input: record đã parse.
    - output: block text đầy đủ gồm Tên/Địa chỉ/Số điện thoại.
    - mục đích của hàm: chuẩn hóa record về dạng text để finetune model.
    """
    name = record.get("name_text", "")
    address = record.get("address_text", "")
    phone = record.get("phone_text", "")

    return (
        f"Tên: {name}\n"
        f"Địa chỉ: {address}\n"
        f"Số điện thoại: {phone}"
    ).strip()


def parse_sample_block(block: str) -> Optional[Dict[str, Any]]:
    """
    - arg/input: 1 block text của 1 sample.
    - output: dict record hoặc None nếu parse lỗi.
    - mục đích của hàm: bóc tách image/name/address/phone từ 1 sample trong file txt.
    """
    block = normalize_text(block)
    if not block:
        return None

    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None

    first_line = lines[0]

    # line đầu thường có dạng: 0.jpg<TAB>Tên: ...
    m = re.match(r"^([^\s]+\.jpg)\s+(.*)$", first_line, flags=re.IGNORECASE)
    if not m:
        return None

    image_name = m.group(1).strip()
    rest_first = m.group(2).strip()

    name_text = ""
    address_text = ""
    phone_text = ""

    # xử lý phần "Tên:" nằm cùng line với image
    if rest_first.startswith("Tên:"):
        name_text = rest_first.replace("Tên:", "", 1).strip()

    for line in lines[1:]:
        if line.startswith("Tên:"):
            name_text = line.replace("Tên:", "", 1).strip()
        elif line.startswith("Địa chỉ:"):
            address_text = line.replace("Địa chỉ:", "", 1).strip()
        elif line.startswith("Số điện thoại:"):
            phone_text = line.replace("Số điện thoại:", "", 1).strip()

    record = {
        "image_name": image_name,
        "name_text": normalize_text(name_text),
        "address_text": normalize_text(address_text),
        "phone_text": normalize_text(phone_text),
    }
    record["full_text"] = record_to_full_text(record)
    return record


def parse_ocr_txt_file(txt_path: str) -> List[Dict[str, Any]]:
    """
    - arg/input: đường dẫn file txt.
    - output: list record đã parse.
    - mục đích của hàm: đọc toàn bộ file txt OCR label và chuyển thành dữ liệu có cấu trúc.
    """
    text = Path(txt_path).read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n")

    # tách block bằng dòng trống
    blocks = re.split(r"\n\s*\n", text)
    records: List[Dict[str, Any]] = []

    for block in blocks:
        record = parse_sample_block(block)
        if record is not None:
            records.append(record)

    return records


def build_address_correction_samples(
    records: List[Dict[str, Any]],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    - arg/input: list record sạch và seed.
    - output: list sample noisy-clean cho task sửa địa chỉ.
    - mục đích của hàm: tạo dataset address-only để model học cấu trúc địa chỉ.
    """
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        clean = rec.get("address_text", "")
        if not clean:
            continue

        noisy = apply_basic_ocr_noise(clean, rng)

        samples.append({
            "id": f"addr_{idx:06d}",
            "task": "address_correction",
            "image_name": rec.get("image_name", ""),
            "input_text": noisy,
            "target_text": clean,
            "domain": "address",
            "source_type": "synthetic_from_clean_label",
            "needs_correction": noisy != clean,
        })

    return samples


def build_full_block_correction_samples(
    records: List[Dict[str, Any]],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    - arg/input: list record sạch và seed.
    - output: list sample noisy-clean cho task sửa block OCR đầy đủ.
    - mục đích của hàm: tạo dataset full block để model học ngữ cảnh giữa các field.
    """
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        clean = rec.get("full_text", "")
        if not clean:
            continue

        noisy_name = apply_basic_ocr_noise(rec.get("name_text", ""), rng)
        noisy_addr = apply_basic_ocr_noise(rec.get("address_text", ""), rng)
        noisy_phone = rec.get("phone_text", "")

        noisy = (
            f"Tên: {noisy_name}\n"
            f"Địa chỉ: {noisy_addr}\n"
            f"Số điện thoại: {noisy_phone}"
        ).strip()

        samples.append({
            "id": f"full_{idx:06d}",
            "task": "full_block_correction",
            "image_name": rec.get("image_name", ""),
            "input_text": normalize_text(noisy),
            "target_text": clean,
            "domain": "address",
            "source_type": "synthetic_from_clean_label",
            "needs_correction": normalize_text(noisy) != clean,
        })

    return samples


def save_jsonl(rows: List[Dict[str, Any]], output_path: str) -> None:
    """
    - arg/input: list dict và đường dẫn output.
    - output: file jsonl.
    - mục đích của hàm: lưu dataset để dùng cho train/eval.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Ví dụ dùng:
    txt_path = "/data/VNPost/Qwen3-Spelling-Error-Correction/data/raw/vlm_results_base_update_final.txt"

    records = parse_ocr_txt_file(txt_path)
    print("Số record:", len(records))
    if records:
        print(records[0])

    address_samples = build_address_correction_samples(records, seed=42)
    full_samples = build_full_block_correction_samples(records, seed=42)

    save_jsonl(address_samples, "./address_correction.jsonl")
    save_jsonl(full_samples, "./full_block_correction.jsonl")