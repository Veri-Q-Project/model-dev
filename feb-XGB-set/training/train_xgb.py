# xgb 학습
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (
    XGB_DATASET_PATH,
    XGB_MODEL_PATH,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    XGB_PARAMS,
)
from feb.feat_schema import FEATURE_COLUMNS


def load_dataset(path=XGB_DATASET_PATH):
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")

    df = pd.read_csv(dataset_path)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"dataset must contain {LABEL_COLUMN!r} column")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].astype(int)

    return X, y


def train_model(X, y):
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("[Evaluation]")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, zero_division=0))


def save_model(model, path=XGB_MODEL_PATH):
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    print(f"[OK] model saved: {model_path}")


def main(dataset_path=XGB_DATASET_PATH, model_path=XGB_MODEL_PATH):
    X, y = load_dataset(dataset_path)

    if len(set(y)) < 2:
        raise ValueError("label must contain both 0 and 1 classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_path)


if __name__ == "__main__":
    main()
