# raw feature를 jsonl로 저장합니다.
# feat_extractor에서 추출한 context를 사용합니다.
# engine의 pipeline에서 사용할 수 있습니다.
import json
from datetime import datetime
from pathlib import Path

from config import RAW_JSONL_PATH, FEB_RAW_SCHEMA_VERSION, DATETIME_TIMESPEC
from feb.feat_extractor import extract_features


def build_raw_row(context, label=None) -> dict:
    url_data = getattr(context, "url", {}) or {}
    redirect_data = getattr(context, "redirect", {}) or {}

    return {
        "schema_version": FEB_RAW_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec=DATETIME_TIMESPEC),
        "label": label,
        "original_url": url_data.get("original_url"),
        "normalized_url": url_data.get("normalized_url"),
        "final_url": redirect_data.get("final_url"),
        "features": extract_features(context),
    }


def append_jsonl(row: dict, path=RAW_JSONL_PATH) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_feb_raw(context, label=None, path=RAW_JSONL_PATH) -> dict:
    row = build_raw_row(context, label=label)
    append_jsonl(row, path=path)
    return row
