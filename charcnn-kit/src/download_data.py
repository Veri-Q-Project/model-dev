# 공개 데이터셋에서 URL 데이터를 다운로드하여 data/raw/urls.csv 생성
#   - 악성: URLhaus (abuse.ch)   https://urlhaus.abuse.ch/downloads/csv_recent/
#   - 정상: Tranco top 1M         https://tranco-list.eu/top-1m.csv.zip
import os
import io
import csv
import random
import zipfile
import urllib.request

import pandas as pd

from config import RAW_CSV_PATH, SEED

URLHAUS_CSV_URL = "https://urlhaus.abuse.ch/downloads/csv_recent/"
TRANCO_ZIP_URL = "https://tranco-list.eu/top-1m.csv.zip"

SAMPLE_PER_CLASS = 5000     # 클래스당 샘플 수 (정상/악성 동일하게 맞춤)
USER_AGENT = "Mozilla/5.0 (charcnn-kit dataset downloader)"
HTTP_TIMEOUT = 60


def _http_get(url: str, binary: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        data = resp.read()
    return data if binary else data.decode("utf-8", errors="replace")


def fetch_malicious() -> list:
    print(f"[malicious] GET {URLHAUS_CSV_URL}")
    text = _http_get(URLHAUS_CSV_URL)

    urls = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = next(csv.reader([line]))
        except (StopIteration, csv.Error):
            continue
        # URLhaus columns: id,dateadded,url,url_status,last_online,threat,tags,urlhaus_link,reporter
        if len(row) < 3 or row[0] == "id":
            continue
        url = row[2].strip()
        if url:
            urls.append(url)

    print(f"[malicious] {len(urls)} URLs parsed")
    return urls


def fetch_benign() -> list:
    print(f"[benign] GET {TRANCO_ZIP_URL}")
    blob = _http_get(TRANCO_ZIP_URL, binary=True)

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        inner_name = zf.namelist()[0]
        with zf.open(inner_name) as f:
            text = f.read().decode("utf-8", errors="replace")

    urls = []
    for line in text.splitlines():
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        domain = parts[1].strip()
        if domain:
            # 도메인만 있으므로 https:// 접두사 부여
            urls.append(f"https://{domain}")

    print(f"[benign] {len(urls)} URLs parsed")
    return urls


def main():
    rng = random.Random(SEED)

    malicious = fetch_malicious()
    benign = fetch_benign()

    if len(malicious) == 0 or len(benign) == 0:
        raise RuntimeError("악성 또는 정상 데이터가 비어있습니다. 네트워크/소스를 확인하세요.")

    # 클래스 균형: 둘 중 작은 쪽 + SAMPLE_PER_CLASS 중 최소
    target = min(SAMPLE_PER_CLASS, len(malicious), len(benign))
    print(f"[balance] sampling {target} per class")

    malicious = rng.sample(malicious, target)
    benign = rng.sample(benign, target)

    rows = [(u, 1) for u in malicious] + [(u, 0) for u in benign]
    rng.shuffle(rows)

    df = pd.DataFrame(rows, columns=["url", "label"])

    os.makedirs(os.path.dirname(RAW_CSV_PATH), exist_ok=True)
    df.to_csv(RAW_CSV_PATH, index=False)

    n = len(df)
    pos = int((df["label"] == 1).sum())
    print(f"saved {n} rows to {RAW_CSV_PATH} (악성 {pos} / 정상 {n - pos})")


if __name__ == "__main__":
    main()
