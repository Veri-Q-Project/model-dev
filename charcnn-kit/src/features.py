# Tabular feature 추출 / 정규화 유틸 (이슈 #3)
#
# - URL 문자열만으로 즉시 계산되는 feature(URL_FEATURE_COLS)는 전처리/추론에서 같은 함수로 계산
# - 페이지 fetch가 필요한 3개(HTML_FEATURE_COLS)는 현재 FEATURE_COLS에서 비활성화
import json
import re
from urllib.parse import parse_qsl, urlparse

import numpy as np

from config import (
    URL_FEATURE_COLS,
    HTML_FEATURE_COLS,
    FEATURE_COLS,
    FEATURE_NORM_PATH,
)

_IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_PERCENT_HEX_RE = re.compile(r"%[0-9a-fA-F]{2}")


def compute_url_features(url: str) -> dict:
    """URL 문자열만으로 계산되는 feature 반환.

    PhiUSIIL과 정확히 동일한 정의를 보장하지는 않지만(공식 정의 미공개),
    preprocess.py와 predict.py가 동일 함수를 쓰므로 학습/추론 일관성은 유지된다.
    """
    url = str(url)
    parsed = urlparse(url if "://" in url else "http://" + url)
    domain = parsed.netloc.split(":")[0]
    parts = [p for p in domain.split(".") if p]
    tld = parts[-1] if parts else ""
    path_segments = [p for p in parsed.path.split("/") if p]
    query_params = parse_qsl(parsed.query, keep_blank_values=True)

    n = len(url)
    n_letters = sum(c.isalpha() for c in url)
    n_digits = sum(c.isdigit() for c in url)
    # "Other special": 알파벳/숫자/일반 URL 구분자가 아닌 모든 문자
    n_other_special = sum(
        1 for c in url
        if not c.isalnum() and c not in "/:?#&=._-"
    )

    return {
        "IsHTTPS": float(parsed.scheme.lower() == "https"),
        "URLLength": float(n),
        "DomainLength": float(len(domain)),
        # 서브도메인 수: "www.x.com" -> 1, "a.b.x.com" -> 2 (최소 0 보장)
        "NoOfSubDomain": float(max(0, len(parts) - 2)),
        "IsDomainIP": float(bool(_IP_RE.match(domain))),
        "TLDLength": float(len(tld)),
        "NoOfLettersInURL": float(n_letters),
        "LetterRatioInURL": round(n_letters / n, 3) if n else 0.0,
        # PhiUSIIL이 사용하는 오타 컬럼명 그대로 유지
        "NoOfDegitsInURL": float(n_digits),
        "DegitRatioInURL": round(n_digits / n, 3) if n else 0.0,
        "SpacialCharRatioInURL": round(n_other_special / n, 3) if n else 0.0,
        "NoOfOtherSpecialCharsInURL": float(n_other_special),
        "HasObfuscation": float(bool(_PERCENT_HEX_RE.search(url))),
        "NoOfEqualsInURL": float(url.count("=")),
        "NoOfQMarkInURL": float(url.count("?")),
        "NoOfAmpersandInURL": float(url.count("&")),
        "NoOfAtInURL": float(url.count("@")),
        "NoOfDashInURL": float(url.count("-")),
        "NoOfDotInURL": float(url.count(".")),
        "NoOfPercentInURL": float(url.count("%")),
        "PathLength": float(len(parsed.path)),
        "QueryLength": float(len(parsed.query)),
        "NoOfPathSegments": float(len(path_segments)),
        "NoOfQueryParams": float(len(query_params)),
        "HasFragment": float(bool(parsed.fragment)),
    }


def save_norm(mean: np.ndarray, std: np.ndarray, path: str = FEATURE_NORM_PATH):
    payload = {
        "cols": list(FEATURE_COLS),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_norm(path: str = FEATURE_NORM_PATH):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload["cols"] != list(FEATURE_COLS):
        raise ValueError(
            "feature 컬럼 구성이 저장 시점과 다릅니다.\n"
            f"  saved={payload['cols']}\n"
            f"  current={list(FEATURE_COLS)}"
        )
    mean = np.array(payload["mean"], dtype="float32")
    std = np.array(payload["std"], dtype="float32")
    return mean, std


def standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """표준화: (x - mean) / std. std=0인 컬럼은 1로 치환해 division-by-zero 방지."""
    std_safe = np.where(std == 0, 1.0, std)
    return ((features - mean) / std_safe).astype("float32")


def assemble_inference_vector(url: str, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """단일 URL 추론용 feature 벡터 생성.

    - URL_FEATURE_COLS: compute_url_features로 직접 계산
    - URL로 계산할 수 없는 active feature: train mean으로 imputation (표준화 후 0)
    """
    url_feats = compute_url_features(url)
    raw = np.zeros(len(FEATURE_COLS), dtype="float32")
    for i, col in enumerate(FEATURE_COLS):
        if col in url_feats:
            raw[i] = url_feats[col]
        else:
            raw[i] = mean[i]
    return standardize(raw, mean, std)
