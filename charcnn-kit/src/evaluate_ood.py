# 저장된 모델을 외부 출처 OOD 데이터셋으로 평가한다.
# CSV는 최소 url,label 컬럼이 필요하다. source 컬럼이 있으면 출처별 오류율도 출력한다.
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from config import (
    BATCH_SIZE,
    DEVICE,
    FEATURE_COLS,
    HARD_EXAMPLES_PATH,
    HTML_FEATURE_COLS,
    MODEL_PATH,
    OOD_CSV_PATH,
    THRESHOLD,
)
from dataset import encode_url, load_vocab
from features import compute_url_features, load_norm, standardize
from model import HybridCharCNN


class OODDataset(Dataset):
    def __init__(self, csv_path: str, vocab: dict, mean: np.ndarray, std: np.ndarray):
        df = pd.read_csv(csv_path)
        if "url" not in df.columns or "label" not in df.columns:
            raise ValueError("OOD CSV에는 url,label 컬럼이 필요합니다.")

        df = df.dropna(subset=["url", "label"]).copy()
        df["url"] = df["url"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)
        df = df[(df["url"].str.len() > 0) & (df["label"].isin([0, 1]))].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("평가할 OOD row가 없습니다.")

        self.df = df
        self.vocab = vocab
        self.mean = mean
        self.std = std
        self.features = self._build_features()

    def _build_features(self) -> np.ndarray:
        rows = []
        for _, row in self.df.iterrows():
            url_feats = compute_url_features(row["url"])
            raw = np.zeros(len(FEATURE_COLS), dtype="float32")

            for i, col in enumerate(FEATURE_COLS):
                if col in url_feats:
                    raw[i] = url_feats[col]
                elif col in self.df.columns and not pd.isna(row[col]):
                    raw[i] = float(row[col])
                else:
                    raw[i] = self.mean[i]

            rows.append(raw)

        features = np.stack(rows)
        return standardize(features, self.mean, self.std)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        url = self.df.iloc[idx]["url"]
        label = self.df.iloc[idx]["label"]

        x = torch.tensor(encode_url(url, self.vocab), dtype=torch.long)
        f = torch.from_numpy(self.features[idx])
        y = torch.tensor(label, dtype=torch.float32)
        return x, f, y


def _metrics(y_true: list[int], y_score: list[float], threshold: float) -> dict:
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _parse_thresholds(raw: str) -> list[float]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"threshold는 0~1 사이여야 합니다: {val}")
        vals.append(val)
    return vals or [THRESHOLD]


def _print_metrics(name: str, m: dict):
    print(f"=== {name} (threshold={m['threshold']:.2f}) ===")
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1 Score : {m['f1']:.4f}")
    print("Confusion Matrix [[TN FP], [FN TP]]:")
    print(f"[[{m['tn']:>5} {m['fp']:>5}]")
    print(f" [{m['fn']:>5} {m['tp']:>5}]]")


def _print_score_summary(df: pd.DataFrame):
    print("\n=== Score Distribution ===")
    group_cols = ["label"]
    if "source" in df.columns:
        group_cols.append("source")

    for key, group in df.groupby(group_cols):
        scores = group["score"].to_numpy()
        q = np.quantile(scores, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
        # group_cols가 1개면 key는 scalar, 2개 이상이면 tuple로 옴 → 항상 tuple로 정규화
        key_tuple = key if isinstance(key, tuple) else (key,)
        label_str = " | ".join(f"{c}={v}" for c, v in zip(group_cols, key_tuple))
        print(
            f"{label_str}: n={len(group)} "
            f"min={q[0]:.4f} p25={q[1]:.4f} med={q[2]:.4f} "
            f"p75={q[3]:.4f} p90={q[4]:.4f} p99={q[5]:.4f} max={q[6]:.4f}"
        )


def _save_hard_examples(df: pd.DataFrame, csv_path: str, output_path: str):
    hard = df[df["label"] != df["pred"]].copy()
    if len(hard) == 0:
        print(f"\n[hard] no hard examples to save ({output_path})")
        return

    hard["error_type"] = np.where(hard["label"] == 1, "false_negative", "false_positive")
    hard["eval_csv"] = csv_path
    hard["threshold"] = THRESHOLD

    keep_cols = ["url", "label", "source", "score", "pred", "error_type", "eval_csv", "threshold"]
    for col in keep_cols:
        if col not in hard.columns:
            hard[col] = ""
    hard = hard[keep_cols]

    if os.path.exists(output_path):
        old = pd.read_csv(output_path)
        hard = pd.concat([old, hard], ignore_index=True)

    hard = hard.dropna(subset=["url", "label"])
    hard["url"] = hard["url"].astype(str).str.strip()
    hard["label"] = hard["label"].astype(int)
    hard = hard.drop_duplicates(subset=["url", "label"], keep="last").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hard.to_csv(output_path, index=False)

    dist = hard["error_type"].value_counts().to_dict()
    print(f"\n[hard] saved {len(hard)} cumulative hard examples to {output_path}: {dist}")


def evaluate_ood(csv_path: str, thresholds: list[float], max_errors: int, save_hard: bool, hard_path: str):
    vocab = load_vocab()
    mean, std = load_norm()

    dataset = OODDataset(csv_path=csv_path, vocab=vocab, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridCharCNN(vocab_size=len(vocab), num_features=len(FEATURE_COLS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true = []
    y_score = []
    with torch.no_grad():
        for x, f, y in loader:
            x = x.to(DEVICE)
            f = f.to(DEVICE)
            logits = model(x, f)
            scores = torch.sigmoid(logits)

            y_true.extend(y.int().tolist())
            y_score.extend(scores.cpu().tolist())

    print(f"loaded {len(dataset)} OOD rows from {csv_path}")
    active_html = [c for c in HTML_FEATURE_COLS if c in FEATURE_COLS]
    if active_html:
        print(f"HTML features without CSV values are imputed to train mean: {active_html}")
    else:
        print("HTML features are inactive; OOD evaluation uses URL-derived features only.")

    print("\n=== Threshold Sweep ===")
    print("thr     acc     prec    recall  f1      FN    FP")
    for threshold in thresholds:
        m = _metrics(y_true, y_score, threshold)
        print(
            f"{threshold:0.3f}  {m['accuracy']:.4f}  {m['precision']:.4f}  "
            f"{m['recall']:.4f}  {m['f1']:.4f}  {m['fn']:>5} {m['fp']:>5}"
        )

    selected = _metrics(y_true, y_score, THRESHOLD)
    print()
    _print_metrics("Selected Config Threshold", selected)

    df = dataset.df.copy()
    df["score"] = y_score
    df["pred"] = (df["score"] >= THRESHOLD).astype(int)

    _print_score_summary(df)

    if "source" in df.columns:
        print("\n=== Source Breakdown ===")
        for source, group in df.groupby("source"):
            n = len(group)
            pos = int((group["label"] == 1).sum())
            fp = int(((group["label"] == 0) & (group["pred"] == 1)).sum())
            fn = int(((group["label"] == 1) & (group["pred"] == 0)).sum())
            print(f"{source}: n={n}, malicious={pos}, FN={fn}, FP={fp}")

    fn_cases = df[(df["label"] == 1) & (df["pred"] == 0)].sort_values("score", ascending=False)
    fp_cases = df[(df["label"] == 0) & (df["pred"] == 1)].sort_values("score", ascending=False)

    print(f"\n=== False Negatives [{len(fn_cases)}건, top {max_errors}] ===")
    for _, row in fn_cases.head(max_errors).iterrows():
        print(f"  score={row['score']:.4f}  {row['url']}")

    print(f"\n=== False Positives [{len(fp_cases)}건, top {max_errors}] ===")
    for _, row in fp_cases.head(max_errors).iterrows():
        print(f"  score={row['score']:.4f}  {row['url']}")

    if save_hard:
        _save_hard_examples(df, csv_path=csv_path, output_path=hard_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=OOD_CSV_PATH)
    parser.add_argument(
        "--thresholds",
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99,0.999",
    )
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--save-hard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-path", default=HARD_EXAMPLES_PATH)
    args = parser.parse_args()

    evaluate_ood(
        csv_path=args.csv,
        thresholds=_parse_thresholds(args.thresholds),
        max_errors=args.max_errors,
        save_hard=args.save_hard,
        hard_path=args.hard_path,
    )


if __name__ == "__main__":
    main()
