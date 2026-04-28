# 원본 url 데이터를 학습/검증/평가용으로 분할
# data/raw/urls.csv -> data/processed/{train,valid,test}.csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RAW_CSV_PATH,
    TRAIN_CSV_PATH,
    VALID_CSV_PATH,
    TEST_CSV_PATH,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    SEED,
    URL_FEATURE_COLS,
    HTML_FEATURE_COLS,
    FEATURE_COLS,
)
from features import compute_url_features


def _validate_ratios():
    total = TRAIN_RATIO + VALID_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"TRAIN+VALID+TEST 비율 합이 1이어야 합니다 (현재: {total})"
        )


def _load_and_clean(raw_path: str) -> pd.DataFrame:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"원본 데이터가 없습니다: {raw_path}\n"
            f"data/raw/urls.csv 파일에 url,label 컬럼으로 데이터를 넣어주세요."
        )

    df = pd.read_csv(raw_path)

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"CSV에 url, label 컬럼이 모두 있어야 합니다 (현재: {list(df.columns)})"
        )

    before = len(df)

    # 1. null 제거
    df = df.dropna(subset=["url", "label"])

    # 2. 타입 정리
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    # 3. label 값 검증 (0/1만 허용)
    invalid = df[~df["label"].isin([0, 1])]
    if len(invalid) > 0:
        raise ValueError(
            f"label은 0(정상) 또는 1(악성)만 허용됩니다. "
            f"잘못된 row {len(invalid)}개 발견."
        )

    # 4. 빈 url 제거
    df = df[df["url"].str.len() > 0]

    # 5. 중복 url 제거 (동일 url에 다른 라벨이 있으면 첫 row만 유지)
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    # 6. URL 기반 feature는 학습/평가/추론 모두 같은 함수 정의를 쓰도록 재계산한다.
    url_features = pd.DataFrame(
        [compute_url_features(url) for url in df["url"]],
        index=df.index,
    )
    for col in URL_FEATURE_COLS:
        df[col] = url_features[col].astype("float32")

    # HTML feature를 FEATURE_COLS에 다시 활성화한 경우, 직접 만든 url,label CSV처럼
    # 값이 없으면 0으로 채워 학습을 계속 진행할 수 있게 한다.
    active_html = [c for c in HTML_FEATURE_COLS if c in FEATURE_COLS]
    missing_html = [c for c in active_html if c not in df.columns]
    if missing_html:
        for col in missing_html:
            df[col] = 0.0
        print(f"[features] missing HTML features filled with 0.0: {missing_html}")

    # 7. tabular feature NaN row 제거 (주로 HTML feature 안전망)
    before_feat = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    if len(df) < before_feat:
        print(f"[clean] feature NaN row {before_feat - len(df)}개 추가 제거")

    after = len(df)
    print(f"[clean] {before} -> {after} rows ({before - after}개 제거)")

    return df


def _stratified_split(df: pd.DataFrame):
    # train vs (valid+test)
    rest_ratio = VALID_RATIO + TEST_RATIO
    train_df, rest_df = train_test_split(
        df,
        test_size=rest_ratio,
        random_state=SEED,
        stratify=df["label"],
    )

    # valid vs test (rest 안에서 비율 유지)
    valid_share = VALID_RATIO / rest_ratio
    valid_df, test_df = train_test_split(
        rest_df,
        test_size=(1 - valid_share),
        random_state=SEED,
        stratify=rest_df["label"],
    )

    return train_df, valid_df, test_df


def _report(name: str, df: pd.DataFrame):
    n = len(df)
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    pos_pct = (pos / n * 100) if n > 0 else 0.0
    print(f"  {name:<6} : {n:>6} rows  (악성 {pos} / 정상 {neg}, 악성비율 {pos_pct:.1f}%)")


def preprocess():
    _validate_ratios()

    df = _load_and_clean(RAW_CSV_PATH)

    if len(df) < 10:
        raise ValueError(
            f"데이터가 너무 적습니다 ({len(df)} rows). "
            f"분할을 위해 최소 10 rows 이상 필요합니다."
        )

    train_df, valid_df, test_df = _stratified_split(df)

    os.makedirs(os.path.dirname(TRAIN_CSV_PATH), exist_ok=True)

    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    valid_df.to_csv(VALID_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)

    print("=== Split Result ===")
    _report("train", train_df)
    _report("valid", valid_df)
    _report("test", test_df)
    print(f"Saved to {os.path.dirname(TRAIN_CSV_PATH)}/")


if __name__ == "__main__":
    preprocess()
