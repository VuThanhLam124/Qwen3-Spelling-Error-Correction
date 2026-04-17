# Mục đích file: đọc file txt OCR label, parse thành record có cấu trúc,
# sinh dataset noisy-clean cho bài toán sửa lỗi địa chỉ/OCR,
# và cho phép cấu hình mức nhiễu bằng file config hoặc CLI.

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Config
# =========================

@dataclass
class SplitConfig:
    """
    - arg/input: tỷ lệ train/dev/test.
    - output: cấu hình split.
    - mục đích của hàm: gom cấu hình chia dữ liệu để tránh leak.
    """
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class NoiseConfig:
    """
    - arg/input: các xác suất và tham số gây nhiễu.
    - output: cấu hình noise.
    - mục đích của hàm: điều khiển augmentation cho dữ liệu địa chỉ/OCR.
    """
    # số bản noisy sinh thêm cho mỗi record sạch
    num_aug_per_record: int = 2

    # tỷ lệ thêm mẫu no-change
    no_change_prob: float = 0.15

    # xác suất áp dụng mất dấu
    partial_drop_prob: float = 0.45
    phrase_drop_prob: float = 0.20
    full_drop_prob: float = 0.08

    # số token mất dấu ở chế độ partial
    partial_min_tokens: int = 1
    partial_max_tokens: int = 3

    # độ dài span ở chế độ phrase
    phrase_min_len: int = 2
    phrase_max_len: int = 4

    # xác suất noise format
    remove_comma_prob: float = 0.25
    remove_colon_prob: float = 0.10
    lowercase_prob: float = 0.08
    compact_space_prob: float = 0.12
    field_label_ascii_prob: float = 0.20

    # xác suất viết tắt đơn vị hành chính
    admin_abbrev_prob: float = 0.30

    # xác suất xóa prefix hành chính
    drop_admin_prefix_prob: float = 0.08


@dataclass
class BuildConfig:
    """
    - arg/input: cấu hình tổng cho build dataset.
    - output: cấu hình build.
    - mục đích của hàm: gom toàn bộ cấu hình để chạy script.
    """
    input_txt: str = "data/raw/ocr_labels.txt"
    output_dir: str = "data/processed/address_dataset"
    seed: int = 42
    split: SplitConfig = SplitConfig()
    noise: NoiseConfig = NoiseConfig()


# =========================
# Utils
# =========================

def normalize_text(text: str) -> str:
    """
    - arg/input: text thô.
    - output: text đã chuẩn hóa unicode và khoảng trắng.
    - mục đích của hàm: làm sạch text trước khi parse và lưu dataset.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    """
    - arg/input: đường dẫn file config json/yaml/yml.
    - output: dict config.
    - mục đích của hàm: đọc cấu hình ngoài để đổi mức nhiễu mà không sửa code.
    """
    if not path:
        return {}

    cfg_path = Path(path)
    text = cfg_path.read_text(encoding="utf-8")

    if cfg_path.suffix.lower() == ".json":
        return json.loads(text)

    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ImportError("Muốn đọc YAML thì cần cài PyYAML.") from e
        return yaml.safe_load(text) or {}

    raise ValueError(f"Chưa hỗ trợ loại config: {cfg_path.suffix}")


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    - arg/input: dict gốc và dict ghi đè.
    - output: dict đã merge.
    - mục đích của hàm: cập nhật config lồng nhau.
    """
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def build_config(args: argparse.Namespace) -> BuildConfig:
    """
    - arg/input: argparse namespace.
    - output: BuildConfig hoàn chỉnh.
    - mục đích của hàm: tạo config cuối cùng từ default + file config + CLI.
    """
    default_cfg = asdict(BuildConfig())
    file_cfg = load_config_file(args.config)
    merged = deep_update(default_cfg, file_cfg)

    if args.input_txt is not None:
        merged["input_txt"] = args.input_txt
    if args.output_dir is not None:
        merged["output_dir"] = args.output_dir
    if args.seed is not None:
        merged["seed"] = args.seed

    if args.num_aug_per_record is not None:
        merged["noise"]["num_aug_per_record"] = args.num_aug_per_record
    if args.no_change_prob is not None:
        merged["noise"]["no_change_prob"] = args.no_change_prob
    if args.partial_drop_prob is not None:
        merged["noise"]["partial_drop_prob"] = args.partial_drop_prob
    if args.phrase_drop_prob is not None:
        merged["noise"]["phrase_drop_prob"] = args.phrase_drop_prob
    if args.full_drop_prob is not None:
        merged["noise"]["full_drop_prob"] = args.full_drop_prob
    if args.admin_abbrev_prob is not None:
        merged["noise"]["admin_abbrev_prob"] = args.admin_abbrev_prob

    return BuildConfig(
        input_txt=merged["input_txt"],
        output_dir=merged["output_dir"],
        seed=merged["seed"],
        split=SplitConfig(**merged["split"]),
        noise=NoiseConfig(**merged["noise"]),
    )


def save_jsonl(rows: List[Dict[str, Any]], output_path: str) -> None:
    """
    - arg/input: list dict và đường dẫn output.
    - output: file jsonl.
    - mục đích của hàm: lưu dataset về dạng jsonl để train/eval.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(x: float) -> float:
    """
    - arg/input: số thực.
    - output: số thực đã chặn vào [0, 1].
    - mục đích của hàm: tránh xác suất cấu hình vượt miền hợp lệ.
    """
    return max(0.0, min(1.0, float(x)))


# =========================
# Parse raw txt
# =========================

def record_to_full_text(record: Dict[str, Any]) -> str:
    """
    - arg/input: record đã parse.
    - output: block text chuẩn.
    - mục đích của hàm: gom các field thành full text dùng cho full-block correction.
    """
    return (
        f"Tên: {record.get('name_text', '')}\n"
        f"Địa chỉ: {record.get('address_text', '')}\n"
        f"Số điện thoại: {record.get('phone_text', '')}"
    ).strip()


def parse_sample_block(block: str) -> Optional[Dict[str, Any]]:
    """
    - arg/input: 1 block text của 1 sample.
    - output: 1 record dict hoặc None.
    - mục đích của hàm: parse image/name/address/phone từ block raw txt.
    """
    block = normalize_text(block)
    if not block:
        return None

    lines = [line.strip() for line in block.split("\n") if line.strip()]
    if not lines:
        return None

    first_line = lines[0]

    # Dạng: 0.jpg<TAB>Tên: ...
    m = re.match(r"^([^\s]+\.jpg)\s+(.*)$", first_line, flags=re.IGNORECASE)
    if not m:
        return None

    image_name = m.group(1).strip()
    rest = m.group(2).strip()

    name_text = ""
    address_text = ""
    phone_text = ""

    if rest.startswith("Tên:"):
        name_text = rest.replace("Tên:", "", 1).strip()

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
    - arg/input: đường dẫn file txt raw.
    - output: list record sạch.
    - mục đích của hàm: đọc file txt OCR label và chuyển thành dữ liệu có cấu trúc.
    """
    text = Path(txt_path).read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text)

    records: List[Dict[str, Any]] = []
    for block in blocks:
        rec = parse_sample_block(block)
        if rec is not None:
            records.append(rec)
    return records


# =========================
# Noise
# =========================

def strip_diacritics(text: str) -> str:
    """
    - arg/input: text có dấu.
    - output: text bỏ dấu.
    - mục đích của hàm: tạo noise mất dấu cho OCR/address.
    """
    if not text:
        return text
    text = text.replace("đ", "d").replace("Đ", "D")
    normalized = unicodedata.normalize("NFD", text)
    no_mark = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", no_mark)


def is_word_token(token: str) -> bool:
    """
    - arg/input: 1 token.
    - output: True/False.
    - mục đích của hàm: chọn token phù hợp để gây nhiễu mà không đụng số hoặc dấu câu.
    """
    if not token:
        return False
    if re.fullmatch(r"\d+", token):
        return False
    if re.fullmatch(r"[^\w]+", token):
        return False
    return any(ch.isalpha() for ch in token)


def tokenize_keep_delimiters(text: str) -> List[str]:
    """
    - arg/input: text.
    - output: list token giữ cả dấu câu và khoảng trắng.
    - mục đích của hàm: sửa token cục bộ nhưng vẫn giữ layout gốc.
    """
    return re.findall(r"\w+|[^\w\s]|\s+", text, flags=re.UNICODE)


def apply_partial_drop(text: str, rng: random.Random, cfg: NoiseConfig) -> str:
    """
    - arg/input: text, rng, cấu hình noise.
    - output: text bị mất dấu ở vài token rời rạc.
    - mục đích của hàm: mô phỏng lỗi OCR cục bộ.
    """
    tokens = tokenize_keep_delimiters(text)
    candidate_ids = [i for i, tok in enumerate(tokens) if is_word_token(tok)]
    if not candidate_ids:
        return text

    max_k = min(cfg.partial_max_tokens, len(candidate_ids))
    min_k = min(cfg.partial_min_tokens, max_k)
    if max_k <= 0:
        return text

    k = rng.randint(min_k, max_k)
    selected = set(rng.sample(candidate_ids, k))
    out = []
    for i, tok in enumerate(tokens):
        out.append(strip_diacritics(tok) if i in selected else tok)
    return "".join(out)


def apply_phrase_drop(text: str, rng: random.Random, cfg: NoiseConfig) -> str:
    """
    - arg/input: text, rng, cấu hình noise.
    - output: text bị mất dấu theo 1 span liên tiếp.
    - mục đích của hàm: mô phỏng OCR fail theo cụm trong địa chỉ.
    """
    tokens = tokenize_keep_delimiters(text)
    word_ids = [i for i, tok in enumerate(tokens) if is_word_token(tok)]
    if not word_ids:
        return text

    span_len = rng.randint(cfg.phrase_min_len, cfg.phrase_max_len)
    if len(word_ids) <= span_len:
        selected = set(word_ids)
    else:
        start_idx = rng.randint(0, len(word_ids) - span_len)
        selected = set(word_ids[start_idx:start_idx + span_len])

    out = []
    for i, tok in enumerate(tokens):
        out.append(strip_diacritics(tok) if i in selected else tok)
    return "".join(out)


def apply_full_drop(text: str) -> str:
    """
    - arg/input: text sạch.
    - output: text mất dấu toàn bộ.
    - mục đích của hàm: tạo một phần nhỏ sample mất dấu hoàn toàn.
    """
    return strip_diacritics(text)


ADMIN_REPLACES = [
    ("Thành phố", "TP."),
    ("thành phố", "TP."),
    ("Tỉnh", "T."),
    ("tỉnh", "t."),
    ("Quận", "Q."),
    ("quận", "Q."),
    ("Huyện", "H."),
    ("huyện", "H."),
    ("Phường", "P."),
    ("phường", "P."),
    ("Xã", "X."),
    ("xã", "X."),
    ("Thị xã", "TX."),
    ("thị xã", "TX."),
    ("Thị trấn", "TT."),
    ("thị trấn", "TT."),
]


def apply_admin_abbrev(text: str, rng: random.Random, cfg: NoiseConfig) -> str:
    """
    - arg/input: text sạch, rng, cấu hình noise.
    - output: text có thể bị viết tắt đơn vị hành chính.
    - mục đích của hàm: dạy model hiểu alias/phân cấp địa chỉ.
    """
    if rng.random() >= cfg.admin_abbrev_prob:
        return text

    out = text
    for src, tgt in ADMIN_REPLACES:
        if src in out and rng.random() < 0.5:
            out = out.replace(src, tgt)
    return out


def apply_drop_admin_prefix(text: str, rng: random.Random, cfg: NoiseConfig) -> str:
    """
    - arg/input: text sạch, rng, cấu hình noise.
    - output: text có thể mất prefix hành chính ở vài chỗ.
    - mục đích của hàm: mô phỏng input nhập thiếu prefix như 'Bắc Ninh' thay vì 'tỉnh Bắc Ninh'.
    """
    if rng.random() >= cfg.drop_admin_prefix_prob:
        return text

    patterns = [
        r"\b[Tt]ỉnh\s+",
        r"\b[Tt]hành phố\s+",
        r"\b[Qq]uận\s+",
        r"\b[Hh]uyện\s+",
        r"\b[Pp]hường\s+",
        r"\b[Xx]ã\s+",
        r"\b[Tt]hị xã\s+",
        r"\b[Tt]hị trấn\s+",
    ]
    out = text
    for p in patterns:
        if re.search(p, out) and rng.random() < 0.4:
            out = re.sub(p, "", out, count=1)
    return out


def apply_format_noise(text: str, rng: random.Random, cfg: NoiseConfig) -> str:
    """
    - arg/input: text, rng, cấu hình noise.
    - output: text bị nhiễu format nhẹ.
    - mục đích của hàm: mô phỏng OCR làm mất dấu câu, space, lowercase.
    """
    out = text

    if rng.random() < cfg.remove_comma_prob:
        out = out.replace(", ", " ")

    if rng.random() < cfg.remove_colon_prob:
        out = out.replace(": ", ":").replace(":", "")

    if rng.random() < cfg.lowercase_prob:
        out = out.lower()

    if rng.random() < cfg.compact_space_prob:
        out = re.sub(r"\s+", " ", out)

    return normalize_text(out)


def apply_field_label_ascii(text: str, rng: random.Random, cfg: NoiseConfig) -> str:
    """
    - arg/input: full block text, rng, cấu hình noise.
    - output: text có thể bị ASCII hóa phần nhãn field.
    - mục đích của hàm: mô phỏng OCR làm sai nhãn 'Tên/Địa chỉ/Số điện thoại'.
    """
    if rng.random() >= cfg.field_label_ascii_prob:
        return text

    out = text
    replaces = {
        "Tên:": "Ten:",
        "Địa chỉ:": "Dia chi:",
        "Số điện thoại:": "So dien thoai:",
    }
    for src, tgt in replaces.items():
        if src in out and rng.random() < 0.7:
            out = out.replace(src, tgt)
    return out


def add_noise_to_address(text: str, rng: random.Random, cfg: NoiseConfig) -> str:
    """
    - arg/input: địa chỉ sạch, rng, cấu hình noise.
    - output: địa chỉ noisy.
    - mục đích của hàm: tạo input lỗi nhưng vẫn giữ semantics địa chỉ.
    """
    out = text

    p_full = safe_float(cfg.full_drop_prob)
    p_phrase = safe_float(cfg.phrase_drop_prob)
    p_partial = safe_float(cfg.partial_drop_prob)

    draw = rng.random()
    if draw < p_full:
        out = apply_full_drop(out)
    elif draw < p_full + p_phrase:
        out = apply_phrase_drop(out, rng, cfg)
    elif draw < p_full + p_phrase + p_partial:
        out = apply_partial_drop(out, rng, cfg)

    out = apply_admin_abbrev(out, rng, cfg)
    out = apply_drop_admin_prefix(out, rng, cfg)
    out = apply_format_noise(out, rng, cfg)
    return normalize_text(out)


def add_noise_to_name(text: str, rng: random.Random, cfg: NoiseConfig) -> str:
    """
    - arg/input: tên sạch, rng, cấu hình noise.
    - output: tên noisy.
    - mục đích của hàm: tạo noise nhẹ cho tên nhưng không phá mạnh semantics.
    """
    out = text

    draw = rng.random()
    if draw < cfg.full_drop_prob * 0.4:
        out = apply_full_drop(out)
    elif draw < cfg.full_drop_prob * 0.4 + cfg.partial_drop_prob * 0.7:
        out = apply_partial_drop(out, rng, cfg)

    out = apply_format_noise(out, rng, cfg)
    return normalize_text(out)


def build_noisy_full_text(record: Dict[str, Any], rng: random.Random, cfg: NoiseConfig) -> str:
    """
    - arg/input: record sạch, rng, cấu hình noise.
    - output: full block noisy.
    - mục đích của hàm: tạo sample full-block correction.
    """
    name = add_noise_to_name(record.get("name_text", ""), rng, cfg)
    address = add_noise_to_address(record.get("address_text", ""), rng, cfg)
    phone = record.get("phone_text", "")

    full_text = (
        f"Tên: {name}\n"
        f"Địa chỉ: {address}\n"
        f"Số điện thoại: {phone}"
    ).strip()

    full_text = apply_field_label_ascii(full_text, rng, cfg)
    full_text = apply_format_noise(full_text, rng, cfg)
    return normalize_text(full_text)


# =========================
# Build dataset
# =========================

def split_records(records: List[Dict[str, Any]], cfg: SplitConfig, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    - arg/input: list record sạch, cấu hình split, seed.
    - output: train/dev/test records.
    - mục đích của hàm: chia dữ liệu ở mức record sạch để tránh leak giữa các bản augment.
    """
    rng = random.Random(seed)
    items = records[:]
    rng.shuffle(items)

    n = len(items)
    n_train = int(n * cfg.train_ratio)
    n_dev = int(n * cfg.dev_ratio)
    n_test = n - n_train - n_dev

    train_records = items[:n_train]
    dev_records = items[n_train:n_train + n_dev]
    test_records = items[n_train + n_dev:n_train + n_dev + n_test]
    return train_records, dev_records, test_records


def build_address_samples(records: List[Dict[str, Any]], split_name: str, seed: int, cfg: NoiseConfig) -> List[Dict[str, Any]]:
    """
    - arg/input: records sạch, tên split, seed, cấu hình noise.
    - output: list sample address-only.
    - mục đích của hàm: tạo dataset sửa lỗi chỉ cho field địa chỉ.
    """
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        clean = rec.get("address_text", "")
        if not clean:
            continue

        # thêm no-change với xác suất nhỏ
        if rng.random() < cfg.no_change_prob:
            rows.append({
                "id": f"{split_name}_addr_clean_{idx:06d}",
                "task": "address_correction",
                "image_name": rec.get("image_name", ""),
                "input_text": clean,
                "target_text": clean,
                "domain": "address",
                "source_type": "no_change",
                "needs_correction": False,
            })

        for j in range(cfg.num_aug_per_record):
            noisy = add_noise_to_address(clean, rng, cfg)
            rows.append({
                "id": f"{split_name}_addr_{idx:06d}_{j}",
                "task": "address_correction",
                "image_name": rec.get("image_name", ""),
                "input_text": noisy,
                "target_text": clean,
                "domain": "address",
                "source_type": "synthetic_address_noise",
                "needs_correction": noisy != clean,
            })

    return rows


def build_full_block_samples(records: List[Dict[str, Any]], split_name: str, seed: int, cfg: NoiseConfig) -> List[Dict[str, Any]]:
    """
    - arg/input: records sạch, tên split, seed, cấu hình noise.
    - output: list sample full-block.
    - mục đích của hàm: tạo dataset sửa lỗi cho block OCR đầy đủ.
    """
    rng = random.Random(seed + 999)
    rows: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        clean = rec.get("full_text", "")
        if not clean:
            continue

        if rng.random() < cfg.no_change_prob:
            rows.append({
                "id": f"{split_name}_full_clean_{idx:06d}",
                "task": "full_block_correction",
                "image_name": rec.get("image_name", ""),
                "input_text": clean,
                "target_text": clean,
                "domain": "address",
                "source_type": "no_change",
                "needs_correction": False,
            })

        for j in range(cfg.num_aug_per_record):
            noisy = build_noisy_full_text(rec, rng, cfg)
            rows.append({
                "id": f"{split_name}_full_{idx:06d}_{j}",
                "task": "full_block_correction",
                "image_name": rec.get("image_name", ""),
                "input_text": noisy,
                "target_text": clean,
                "domain": "address",
                "source_type": "synthetic_full_block_noise",
                "needs_correction": noisy != clean,
            })

    return rows


def save_split(
    split_name: str,
    records: List[Dict[str, Any]],
    output_dir: str,
    seed: int,
    noise_cfg: NoiseConfig,
) -> Dict[str, int]:
    """
    - arg/input: tên split, records sạch, thư mục output, seed, noise config.
    - output: thống kê số dòng đã lưu.
    - mục đích của hàm: build và lưu toàn bộ file jsonl cho một split.
    """
    address_rows = build_address_samples(records, split_name, seed, noise_cfg)
    full_rows = build_full_block_samples(records, split_name, seed, noise_cfg)

    save_jsonl(records, str(Path(output_dir) / f"{split_name}_canonical_records.jsonl"))
    save_jsonl(address_rows, str(Path(output_dir) / f"{split_name}_address_correction.jsonl"))
    save_jsonl(full_rows, str(Path(output_dir) / f"{split_name}_full_block_correction.jsonl"))

    return {
        "canonical_records": len(records),
        "address_rows": len(address_rows),
        "full_rows": len(full_rows),
    }


def save_build_meta(cfg: BuildConfig, output_dir: str, stats: Dict[str, Any]) -> None:
    """
    - arg/input: config build, output_dir, thống kê.
    - output: file json metadata.
    - mục đích của hàm: lưu lại cấu hình và thống kê để tái lập dataset.
    """
    meta = {
        "input_txt": cfg.input_txt,
        "seed": cfg.seed,
        "split": asdict(cfg.split),
        "noise": asdict(cfg.noise),
        "stats": stats,
    }
    out = Path(output_dir) / "build_meta.json"
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    """
    - arg/input: không có.
    - output: argparse namespace.
    - mục đích của hàm: nhận tham số chạy script từ CLI.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None, help="file config json/yaml/yml")
    parser.add_argument("--input_txt", type=str, default=None, help="đường dẫn file txt raw")
    parser.add_argument("--output_dir", type=str, default=None, help="thư mục lưu dataset")
    parser.add_argument("--seed", type=int, default=None, help="seed ngẫu nhiên")

    parser.add_argument("--num_aug_per_record", type=int, default=None, help="số bản noisy sinh thêm mỗi record")
    parser.add_argument("--no_change_prob", type=float, default=None, help="xác suất thêm mẫu no-change")
    parser.add_argument("--partial_drop_prob", type=float, default=None, help="xác suất mất dấu cục bộ")
    parser.add_argument("--phrase_drop_prob", type=float, default=None, help="xác suất mất dấu theo cụm")
    parser.add_argument("--full_drop_prob", type=float, default=None, help="xác suất mất dấu toàn câu")
    parser.add_argument("--admin_abbrev_prob", type=float, default=None, help="xác suất viết tắt đơn vị hành chính")

    return parser.parse_args()


def main() -> None:
    """
    - arg/input: không có.
    - output: file train/dev/test jsonl trong output_dir.
    - mục đích của hàm: chạy toàn bộ pipeline build dataset từ file txt raw.
    """
    args = parse_args()
    cfg = build_config(args)

    records = parse_ocr_txt_file(cfg.input_txt)
    if not records:
        raise ValueError("Không parse được record nào từ file txt.")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records, dev_records, test_records = split_records(records, cfg.split, cfg.seed)

    stats = {
        "total_records": len(records),
        "train": save_split("train", train_records, str(output_dir), cfg.seed, cfg.noise),
        "dev": save_split("dev", dev_records, str(output_dir), cfg.seed + 1, cfg.noise),
        "test": save_split("test", test_records, str(output_dir), cfg.seed + 2, cfg.noise),
    }

    save_build_meta(cfg, str(output_dir), stats)

    print("=== Build xong ===")
    print("input_txt:", cfg.input_txt)
    print("output_dir:", cfg.output_dir)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()