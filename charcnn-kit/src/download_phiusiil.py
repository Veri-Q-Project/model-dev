# UCI ML Repository에서 PhiUSIIL Phishing URL Dataset 다운로드
#   출처: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
#   235k URL, 정상/악성이 비슷한 형태로 정제된 학술 데이터셋
#
# PhiUSIIL의 label 규칙:  1 = legitimate, 0 = phishing
# 본 프로젝트의 label 규칙: 0 = 정상,      1 = 악성
# -> 다운로드 후 라벨 반전 처리
import os
import io
import zipfile
import urllib.request

import pandas as pd

from config import RAW_CSV_PATH, SEED

PHIUSIIL_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/967/"
    "phiusiil+phishing+url+dataset.zip"
)
USER_AGENT = "Mozilla/5.0 (charcnn-kit dataset downloader)"
HTTP_TIMEOUT = 180

# 클래스 합산 최대 샘플 수 (None이면 전체 사용)
# CharCNN CPU 학습 시간을 고려하여 5만으로 제한
MAX_SAMPLES = 50000


def _http_get_binary(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return resp.read()


def _find_url_col(cols):
    for c in cols:
        if c.lower() == "url":
            return c
    return None


def main():
    print(f"GET {PHIUSIIL_ZIP_URL}")
    blob = _http_get_binary(PHIUSIIL_ZIP_URL)
    print(f"downloaded {len(blob) / 1024 / 1024:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = zf.namelist()
        csv_name = next((n for n in names if n.lower().endswith(".csv")), None)
        if csv_name is None:
            raise RuntimeError(f"zip 안에 CSV가 없습니다: {names}")
        print(f"reading {csv_name}")
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    print(f"loaded {len(df)} rows, {len(df.columns)} columns")

    url_col = _find_url_col(df.columns)
    if url_col is None or "label" not in df.columns:
        raise RuntimeError(
            f"url/label 컬럼을 찾지 못했습니다. 가용: {list(df.columns)}"
        )
    print(f"using url_col='{url_col}', label_col='label'")

    # 라벨 분포 확인 (PhiUSIIL: 1=legit, 0=phish)
    dist = df["label"].value_counts().to_dict()
    print(f"PhiUSIIL label dist (raw): {dist}")

    df_out = df[[url_col, "label"]].copy()
    df_out.columns = ["url", "label"]

    # 라벨 반전: PhiUSIIL(1=legit, 0=phish) -> 본 프로젝트(0=정상, 1=악성)
    df_out["label"] = 1 - df_out["label"].astype(int)

    print("flipped label dist (project): "
          f"{df_out['label'].value_counts().to_dict()}")

    # 클래스별 균형 샘플링
    if MAX_SAMPLES is not None and len(df_out) > MAX_SAMPLES:
        per_class = MAX_SAMPLES // 2
        pos = df_out[df_out["label"] == 1]
        neg = df_out[df_out["label"] == 0]

        pos_n = min(per_class, len(pos))
        neg_n = min(per_class, len(neg))

        sampled = pd.concat([
            pos.sample(n=pos_n, random_state=SEED),
            neg.sample(n=neg_n, random_state=SEED),
        ])
        df_out = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(RAW_CSV_PATH), exist_ok=True)
    df_out.to_csv(RAW_CSV_PATH, index=False)

    n = len(df_out)
    pos = int((df_out["label"] == 1).sum())
    print(f"saved {n} rows to {RAW_CSV_PATH} (악성 {pos} / 정상 {n - pos})")


if __name__ == "__main__":
    main()
