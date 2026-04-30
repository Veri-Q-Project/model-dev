# config.py
# FEB + XGB 프로젝트 전역 설정 파일입니다.
# 경로, 모델 설정, 학습 파라미터, 라벨 문자열처럼 여러 파일에서 공유되는 값을 한 곳에서 관리합니다.

from pathlib import Path


# =========================
# Base paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" # 전반적인 결과들
MODEL_DIR = BASE_DIR / "models" # 서비스용, 최종 확인된 모델


# =========================
# Dataset paths
# =========================
RAW_JSONL_PATH = RAW_DATA_DIR / "feb_raw.jsonl"
XGB_DATASET_PATH = PROCESSED_DATA_DIR / "xgb_dataset.csv"


# =========================
# Model paths
# =========================
XGB_MODEL_PATH = OUTPUT_DIR / "xgb_model.joblib"


# =========================
# Raw export settings
# =========================
FEB_RAW_SCHEMA_VERSION = "feb_raw_v1"
DATETIME_TIMESPEC = "seconds"


# =========================
# Inference settings
# =========================
XGB_THRESHOLD = 0.5
HIGH_RISK_SCORE_THRESHOLD = 80

XGB_THREAT_LABEL = "xgb_suspicious_url"
XGB_HIGH_RISK_THREAT_LABEL = "xgb_high_risk_url"


# =========================
# Train settings
# =========================
TEST_SIZE = 0.2
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
}


# =========================
# Dataset column settings
# =========================
LABEL_COLUMN = "label"
URL_META_COLUMNS = [
    "original_url",
    "normalized_url",
    "final_url",
]


# =========================
# Feature keyword settings
# =========================
SENSITIVE_KEYWORDS = [
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "confirm",
    "admin",
    "signin",
    "password",
    "bank",
    "payment",
]

FREE_HOSTING_KEYWORDS = [
    "000webhost",
    "github.io",
    "pages.dev",
    "netlify.app",
    "vercel.app",
    "firebaseapp.com",
    "web.app",
]

SUSPICIOUS_EXTENSIONS = [
    ".exe",
    ".zip",
    ".apk",
    ".scr",
    ".bat",
    ".cmd",
    ".msi",
]
