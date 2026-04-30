# inference/xgb_predictor.py
import pandas as pd

from config import (
    XGB_THRESHOLD,
    HIGH_RISK_SCORE_THRESHOLD,
    XGB_THREAT_LABEL,
    XGB_HIGH_RISK_THREAT_LABEL,
)
from feb.feat_schema import FEATURE_COLUMNS
from inference.loader import load_xgb_model


class XGBPredictor:
    def __init__(self, model_path=None, threshold: float = XGB_THRESHOLD):
        if model_path:
            self.model = load_xgb_model(model_path)
        else:
            self.model = load_xgb_model()

        self.threshold = threshold

    def predict_from_features(self, features: dict) -> dict:
        row = {column: features.get(column, 0) for column in FEATURE_COLUMNS}
        X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        probability = float(self.model.predict_proba(X)[0][1])
        label = int(probability >= self.threshold)
        score = round(probability * 100)

        return {
            "label": label,
            "score": score,
            "probability": probability,
            "threats": self._build_threats(label, score),
        }

    def _build_threats(self, label: int, score: int) -> list[str]:
        if label == 0:
            return []

        threats = [XGB_THREAT_LABEL]

        if score >= HIGH_RISK_SCORE_THRESHOLD:
            threats.append(XGB_HIGH_RISK_THREAT_LABEL)

        return threats
