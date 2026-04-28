# 외부 출처 OOD 평가셋 생성
#   - 악성: URLhaus recent
#   - 정상: Tranco top 1M
#
# 학습용 data/raw/urls.csv를 덮어쓰지 않고 data/ood/ood_test.csv에 저장한다.
import argparse
import os
import random

import pandas as pd

from config import OOD_BENIGN_TOP_N, OOD_CSV_PATH, OOD_SAMPLE_PER_CLASS, RAW_CSV_PATH, SEED
from download_data import fetch_benign, fetch_malicious
from download_train_mixed import fetch_openphish, fetch_phishtank, _safe_fetch


def _dedupe(urls: list[str]) -> list[str]:
    seen = set()
    out = []
    for url in urls:
        key = str(url).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _load_excluded_urls(path: str) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    df = pd.read_csv(path, usecols=["url"])
    excluded = set(df["url"].dropna().astype(str).str.strip())
    print(f"[exclude] loaded {len(excluded)} URLs from {path}")
    return excluded


def build_ood(csv_path: str, sample_per_class: int, benign_top_n: int, exclude_csv: str):
    rng = random.Random(SEED)

    urlhaus = _dedupe(fetch_malicious())
    openphish = _dedupe(_safe_fetch("openphish", fetch_openphish))
    phishtank = _dedupe(_safe_fetch("phishtank", fetch_phishtank))
    malicious_rows = (
        [{"url": url, "source": "urlhaus_recent"} for url in urlhaus]
        + [{"url": url, "source": "openphish"} for url in openphish]
        + [{"url": url, "source": "phishtank"} for url in phishtank]
    )
    benign = _dedupe(fetch_benign())
    if benign_top_n > 0:
        benign = benign[:benign_top_n]
        print(f"[benign] limited to top {benign_top_n} Tranco domains")

    excluded = _load_excluded_urls(exclude_csv)
    if excluded:
        before_mal = len(malicious_rows)
        before_ben = len(benign)
        malicious_rows = [row for row in malicious_rows if row["url"] not in excluded]
        benign = [url for url in benign if url not in excluded]
        print(
            f"[exclude] removed train overlap: "
            f"malicious {before_mal - len(malicious_rows)}, benign {before_ben - len(benign)}"
        )

    malicious_rows = list({row["url"]: row for row in malicious_rows}.values())
    malicious = [row["url"] for row in malicious_rows]
    malicious_set = set(malicious)
    benign = [url for url in benign if url not in malicious_set]

    if not malicious_rows or not benign:
        raise RuntimeError("OOD 평가셋을 만들 수 없습니다. 외부 데이터 소스를 확인하세요.")

    target = min(sample_per_class, len(malicious_rows), len(benign))
    if target < sample_per_class:
        print(
            f"[warn] requested {sample_per_class} per class, "
            f"but only {target} can be sampled"
        )

    malicious_rows = rng.sample(malicious_rows, target)
    benign = rng.sample(benign, target)

    rows = (
        [{"url": row["url"], "label": 1, "source": row["source"]} for row in malicious_rows]
        + [{"url": url, "label": 0, "source": "tranco_top"} for url in benign]
    )
    rng.shuffle(rows)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(
        f"saved {len(df)} rows to {csv_path} "
        f"(악성 {target} / 정상 {target})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=OOD_CSV_PATH)
    parser.add_argument("--sample-per-class", type=int, default=OOD_SAMPLE_PER_CLASS)
    parser.add_argument("--benign-top-n", type=int, default=OOD_BENIGN_TOP_N)
    parser.add_argument("--exclude-csv", default=RAW_CSV_PATH)
    args = parser.parse_args()

    build_ood(
        csv_path=args.csv,
        sample_per_class=args.sample_per_class,
        benign_top_n=args.benign_top_n,
        exclude_csv=args.exclude_csv,
    )


if __name__ == "__main__":
    main()
