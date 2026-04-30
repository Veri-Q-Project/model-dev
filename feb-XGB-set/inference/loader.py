# inference/model_loader.py
from pathlib import Path

import joblib

from config import XGB_MODEL_PATH


def load_xgb_model(model_path=XGB_MODEL_PATH):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    return joblib.load(path)
