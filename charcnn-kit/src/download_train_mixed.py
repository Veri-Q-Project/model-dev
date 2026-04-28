# 혼합 학습 데이터셋 생성
#   - 악성: PhiUSIIL phishing + URLhaus recent
#           + OpenPhish community + PhishTank online-valid
#   - 정상: PhiUSIIL legitimate + Tranco top domains
#
# 결과는 data/raw/urls.csv에 저장한다. label 규칙은 0=정상, 1=악성.
import argparse
import bz2
import csv
import hashlib
import io
import os
import random
import zipfile
import urllib.parse
import urllib.request

import pandas as pd

from config import (
    DYNAMIC_AUG_PER_CLASS,
    HARD_EXAMPLES_MAX_PER_CLASS,
    HARD_EXAMPLES_PATH,
    MIXED_OPENPHISH_SAMPLES,
    MIXED_PHISHTANK_SAMPLES,
    MIXED_PHIUSIIL_PER_CLASS,
    MIXED_TRANCO_SAMPLES,
    MIXED_TRANCO_TOP_N,
    MIXED_URLHAUS_SAMPLES,
    RAW_CSV_PATH,
    SEED,
)
from download_data import fetch_benign, fetch_malicious
from download_phiusiil import PHIUSIIL_ZIP_URL, USER_AGENT, _find_url_col

OPENPHISH_FEED_URL = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
PHISHTANK_PUBLIC_CSV_BZ2_URL = "http://data.phishtank.com/data/online-valid.csv.bz2"
PHISHTANK_KEYED_CSV_BZ2_URL = "http://data.phishtank.com/data/{key}/online-valid.csv.bz2"
PHISHTANK_APP_KEY_ENV = "PHISHTANK_APP_KEY"


def _http_get_binary(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def _http_get_text(url: str) -> str:
    return _http_get_binary(url).decode("utf-8", errors="replace")


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


def _sample(urls: list[str], n: int, rng: random.Random, name: str) -> list[str]:
    urls = _dedupe(urls)
    target = min(n, len(urls))
    if target < n:
        print(f"[warn] {name}: requested {n}, available {target}")
    return rng.sample(urls, target)


def _safe_fetch(name: str, fetcher) -> list[str]:
    try:
        return fetcher()
    except Exception as exc:
        print(f"[warn] {name} fetch failed: {exc}")
        return []


def fetch_phiusiil_urls(per_class: int, rng: random.Random) -> tuple[list[str], list[str]]:
    print(f"[phiusiil] GET {PHIUSIIL_ZIP_URL}")
    blob = _http_get_binary(PHIUSIIL_ZIP_URL)
    print(f"[phiusiil] downloaded {len(blob) / 1024 / 1024:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if csv_name is None:
            raise RuntimeError("PhiUSIIL zip 안에 CSV가 없습니다.")
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    url_col = _find_url_col(df.columns)
    if url_col is None or "label" not in df.columns:
        raise RuntimeError(
            f"PhiUSIIL CSV에 url/label 컬럼을 찾지 못했습니다: {list(df.columns)}"
        )

    # PhiUSIIL raw label: 1=legitimate, 0=phishing
    phish = df[df["label"] == 0][url_col].astype(str).tolist()
    legit = df[df["label"] == 1][url_col].astype(str).tolist()

    phish = _sample(phish, per_class, rng, "phiusiil_phish")
    legit = _sample(legit, per_class, rng, "phiusiil_legit")
    print(f"[phiusiil] sampled phishing={len(phish)}, legitimate={len(legit)}")
    return phish, legit


def fetch_openphish() -> list[str]:
    print(f"[openphish] GET {OPENPHISH_FEED_URL}")
    text = _http_get_text(OPENPHISH_FEED_URL)
    urls = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    print(f"[openphish] {len(urls)} URLs parsed")
    return urls


def fetch_phishtank() -> list[str]:
    key = os.environ.get(PHISHTANK_APP_KEY_ENV, "").strip()
    if key:
        url = PHISHTANK_KEYED_CSV_BZ2_URL.format(key=urllib.parse.quote(key, safe=""))
    else:
        url = PHISHTANK_PUBLIC_CSV_BZ2_URL
    key_note = "with app key" if key else "without app key"
    print(f"[phishtank] GET {url} ({key_note})")
    blob = _http_get_binary(url)
    text = bz2.decompress(blob).decode("utf-8", errors="replace")

    urls = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        if row.get("verified", "yes").lower() != "yes":
            continue
        if row.get("online", "yes").lower() != "yes":
            continue
        value = (row.get("url") or "").strip()
        if value:
            urls.append(value)

    print(f"[phishtank] {len(urls)} URLs parsed")
    return urls


def _token(seed: str, length: int = 12) -> str:
    return hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _append_query(url: str, params: list[tuple[str, str]]) -> str:
    joiner = "&" if "?" in url else "?"
    query = "&".join(f"{k}={v}" for k, v in params)
    return f"{url}{joiner}{query}"


def _append_path(url: str, segment: str) -> str:
    if "?" in url:
        base, query = url.split("?", 1)
        suffix = "?" + query
    else:
        base, suffix = url, ""
    base = base.rstrip("/")
    return f"{base}/{segment}{suffix}"


def _dynamic_variant(url: str, label: int, idx: int, rng: random.Random) -> str:
    token = _token(f"{url}|{idx}|{label}")
    if label == 0:
        campaign = rng.choice(["spring", "newsletter", "catalog", "app", "promo"])
        mode = rng.randrange(6)
        if mode == 0:
            return _append_query(url, [("utm_source", "mail"), ("utm_campaign", campaign)])
        if mode == 1:
            return _append_query(url, [("page", str((idx % 20) + 1)), ("sort", "recent")])
        if mode == 2:
            return _append_path(url, f"products/{int(token[:6], 16) % 1000000}")
        if mode == 3:
            return _append_query(_append_path(url, "search"), [("q", "support"), ("lang", "en")])
        if mode == 4:
            return _append_path(url, f"news/2026/{(idx % 12) + 1:02d}/{token[:6]}")
        return _append_query(url, [("ref", token[:8]), ("view", "mobile")])

    mode = rng.randrange(6)
    if mode == 0:
        return _append_path(url, f"login/{token[:10]}")
    if mode == 1:
        return _append_path(url, f"verify/account/{token[:8]}")
    if mode == 2:
        target = "https%3A%2F%2Fexample.com%2Fcontinue"
        return _append_query(url, [("next", target), ("id", token[:10])])
    if mode == 3:
        return _append_query(_append_path(url, "secure/update"), [("session", token)])
    if mode == 4:
        return _append_path(url, f"invoice/{int(token[:8], 16) % 100000000}")
    return _append_query(_append_path(url, f"u/{token[:6]}"), [("token", token)])


def add_dynamic_augments(df: pd.DataFrame, per_class: int, rng: random.Random) -> pd.DataFrame:
    if per_class <= 0:
        return df

    rows = []
    for label in [0, 1]:
        source = df[df["label"] == label]
        target = min(per_class, len(source))
        if target <= 0:
            continue
        sampled = source.sample(n=target, random_state=SEED + label)
        for i, (_, row) in enumerate(sampled.iterrows()):
            rows.append({
                "url": _dynamic_variant(str(row["url"]), label, i, rng),
                "label": label,
                "source": f"dynamic_aug_{'malicious' if label else 'benign'}",
            })

    if not rows:
        return df

    aug_df = pd.DataFrame(rows)
    print(
        f"[dynamic] added {len(aug_df)} synthetic dynamic URLs "
        f"({int((aug_df['label'] == 1).sum())} malicious / "
        f"{int((aug_df['label'] == 0).sum())} benign)"
    )
    return pd.concat([df, aug_df], ignore_index=True)


def load_hard_examples(path: str, max_per_class: int, rng: random.Random) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        print(f"[hard] no hard examples found: {path}")
        return pd.DataFrame(columns=["url", "label", "source"])

    df = pd.read_csv(path)
    if "url" not in df.columns or "label" not in df.columns:
        print(f"[hard] ignored invalid hard example CSV: {path}")
        return pd.DataFrame(columns=["url", "label", "source"])

    df = df.dropna(subset=["url", "label"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[(df["url"].str.len() > 0) & (df["label"].isin([0, 1]))]
    df = df.drop_duplicates(subset=["url", "label"], keep="last")

    pieces = []
    for label in [0, 1]:
        part = df[df["label"] == label]
        target = min(max_per_class, len(part))
        if target > 0:
            pieces.append(part.sample(n=target, random_state=SEED + 100 + label))

    if not pieces:
        print(f"[hard] no usable hard examples in {path}")
        return pd.DataFrame(columns=["url", "label", "source"])

    out = pd.concat(pieces, ignore_index=True)
    if "error_type" in out.columns:
        out["source"] = out["error_type"].map({
            "false_negative": "hard_false_negative",
            "false_positive": "hard_false_positive",
        }).fillna("hard_example")
    else:
        out["source"] = "hard_example"

    out = out[["url", "label", "source"]].sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(
        f"[hard] loaded {len(out)} hard examples from {path} "
        f"(악성 {int((out['label'] == 1).sum())} / 정상 {int((out['label'] == 0).sum())})"
    )
    return out


def _rebalance_preserving_hard(df: pd.DataFrame) -> pd.DataFrame:
    hard_mask = df["source"].astype(str).str.startswith("hard_")
    hard_df = df[hard_mask]
    base_df = df[~hard_mask]

    pos_total = int((df["label"] == 1).sum())
    neg_total = int((df["label"] == 0).sum())
    target = min(pos_total, neg_total)

    pieces = [hard_df]
    for label in [0, 1]:
        hard_n = int((hard_df["label"] == label).sum())
        needed = max(0, target - hard_n)
        part = base_df[base_df["label"] == label]
        if len(part) <= needed:
            pieces.append(part)
        else:
            pieces.append(part.sample(n=needed, random_state=SEED + 200 + label))

    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_mixed_train(
    output: str,
    phiusiil_per_class: int,
    urlhaus_samples: int,
    openphish_samples: int,
    phishtank_samples: int,
    tranco_samples: int,
    dynamic_aug_per_class: int,
    hard_examples_path: str,
    hard_max_per_class: int,
    tranco_top_n: int,
):
    rng = random.Random(SEED)

    phi_phish, phi_legit = fetch_phiusiil_urls(phiusiil_per_class, rng)

    urlhaus = fetch_malicious()
    openphish = _safe_fetch("openphish", fetch_openphish)
    phishtank = _safe_fetch("phishtank", fetch_phishtank)
    tranco = fetch_benign()
    if tranco_top_n > 0:
        tranco = tranco[:tranco_top_n]
        print(f"[tranco] limited to top {tranco_top_n} domains")

    # 정상 후보에서 악성 URL과 정확히 같은 항목은 제거
    malicious_seen = set(_dedupe(urlhaus + openphish + phishtank + phi_phish))
    tranco = [url for url in tranco if url not in malicious_seen]
    phi_legit = [url for url in phi_legit if url not in malicious_seen]

    urlhaus = _sample(urlhaus, urlhaus_samples, rng, "urlhaus")
    openphish = _sample(openphish, openphish_samples, rng, "openphish")
    phishtank = _sample(phishtank, phishtank_samples, rng, "phishtank")
    tranco = _sample(tranco, tranco_samples, rng, "tranco")

    rows = (
        [{"url": url, "label": 1, "source": "phiusiil_phish"} for url in phi_phish]
        + [{"url": url, "label": 0, "source": "phiusiil_legit"} for url in phi_legit]
        + [{"url": url, "label": 1, "source": "urlhaus_recent"} for url in urlhaus]
        + [{"url": url, "label": 1, "source": "openphish"} for url in openphish]
        + [{"url": url, "label": 1, "source": "phishtank"} for url in phishtank]
        + [{"url": url, "label": 0, "source": "tranco_top"} for url in tranco]
    )
    rng.shuffle(rows)

    df = pd.DataFrame(rows).drop_duplicates(subset=["url"], keep="first")
    # 중복 제거 후 클래스 균형 재조정
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    target = min(len(pos), len(neg))
    df = pd.concat([
        pos.sample(n=target, random_state=SEED),
        neg.sample(n=target, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df = add_dynamic_augments(df, dynamic_aug_per_class, rng)
    hard_df = load_hard_examples(hard_examples_path, hard_max_per_class, rng)
    if len(hard_df) > 0:
        df = pd.concat([df, hard_df], ignore_index=True)
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    df = _rebalance_preserving_hard(df)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)

    print(
        f"saved {len(df)} rows to {output} "
        f"(악성 {int((df['label'] == 1).sum())} / 정상 {int((df['label'] == 0).sum())})"
    )
    print("source distribution:")
    print(df["source"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=RAW_CSV_PATH)
    parser.add_argument("--phiusiil-per-class", type=int, default=MIXED_PHIUSIIL_PER_CLASS)
    parser.add_argument("--urlhaus-samples", type=int, default=MIXED_URLHAUS_SAMPLES)
    parser.add_argument("--openphish-samples", type=int, default=MIXED_OPENPHISH_SAMPLES)
    parser.add_argument("--phishtank-samples", type=int, default=MIXED_PHISHTANK_SAMPLES)
    parser.add_argument("--tranco-samples", type=int, default=MIXED_TRANCO_SAMPLES)
    parser.add_argument("--dynamic-aug-per-class", type=int, default=DYNAMIC_AUG_PER_CLASS)
    parser.add_argument("--hard-examples", default=HARD_EXAMPLES_PATH)
    parser.add_argument("--hard-max-per-class", type=int, default=HARD_EXAMPLES_MAX_PER_CLASS)
    parser.add_argument("--tranco-top-n", type=int, default=MIXED_TRANCO_TOP_N)
    args = parser.parse_args()

    build_mixed_train(
        output=args.output,
        phiusiil_per_class=args.phiusiil_per_class,
        urlhaus_samples=args.urlhaus_samples,
        openphish_samples=args.openphish_samples,
        phishtank_samples=args.phishtank_samples,
        tranco_samples=args.tranco_samples,
        dynamic_aug_per_class=args.dynamic_aug_per_class,
        hard_examples_path=args.hard_examples,
        hard_max_per_class=args.hard_max_per_class,
        tranco_top_n=args.tranco_top_n,
    )


if __name__ == "__main__":
    main()
