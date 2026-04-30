# jsonl -> XGB 학습용 CSV 변환기
import csv
import json
from pathlib import Path

from config import RAW_JSONL_PATH, XGB_DATASET_PATH, LABEL_COLUMN, URL_META_COLUMNS
from feb.feat_schema import FEATURE_COLUMNS


def load_jsonl(path=RAW_JSONL_PATH) -> list[dict]:
    rows = []
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"raw jsonl file not found: {path}")

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def build_dataset_rows(raw_rows: list[dict]) -> list[dict]:
    dataset_rows = []

    for raw in raw_rows:
        label = raw.get(LABEL_COLUMN)

        # label 없는 데이터는 학습용 CSV에서 제외
        if label is None:
            continue

        features = raw.get("features", {})

        row = {LABEL_COLUMN: int(label)}

        for meta_column in URL_META_COLUMNS:
            row[meta_column] = raw.get(meta_column, "")

        for column in FEATURE_COLUMNS:
            row[column] = features.get(column, 0)

        dataset_rows.append(row)

    return dataset_rows


def save_csv(rows: list[dict], output_path=XGB_DATASET_PATH) -> None:
    save_path = Path(output_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        LABEL_COLUMN,
        *URL_META_COLUMNS,
        *FEATURE_COLUMNS,
    ]

    with save_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(raw_path=RAW_JSONL_PATH, output_path=XGB_DATASET_PATH) -> None:
    raw_rows = load_jsonl(raw_path)
    dataset_rows = build_dataset_rows(raw_rows)
    save_csv(dataset_rows, output_path)

    print(f"[OK] raw rows: {len(raw_rows)}")
    print(f"[OK] labeled rows: {len(dataset_rows)}")
    print(f"[OK] saved: {output_path}")


if __name__ == "__main__":
    build_dataset()
