# 설정값 관리
import torch

MAX_LEN = 200               # URL 최대 길이
BATCH_SIZE = 32             # 배치 크기: 한번에 몇 개 데이터를 묶어 학습할것인가
EPOCHS = 20                 # 최대 에폭 수: early stopping이 실제 종료를 결정함
PATIENCE = 3                # early stopping: valid_loss가 N epoch 동안 개선 없으면 중단
LEARNING_RATE = 0.001       # 학습률: 한 번 업데이트 시 얼마나 수정을 거칠지
EMBED_DIM = 64              # 문자 임베딩 차원 수: 각 문자를 몇 개 숫자로 표현할지
NUM_FILTERS = 128           # CNN 필터 수 = url 패턴 찾는 탐지기 수 (너무 많을시 과적합 발생 가능)
KERNEL_SIZES = [3, 4, 5]    # 패턴 크기 목록(필터): 3글자(예: log), 4글자(예: bank), 5글자(예: login)
DROPOUT = 0.3               # 드롭아웃 비율: 학습 중 몇%의 뉴련을 쉬게 할지 (과적합 방지)

# GPU 있으면 GPU 사용-> 없으면 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pridict
THRESHOLD = 0.2             # score가 0.2 이상이면 악성으로 판단 (FN/FP 균형)

VOCAB_PATH = "saved/char_vocab.json" # 문자 사전 저장 경로
MODEL_PATH = "saved/charcnn.pt"      # 모델 저장 경로

# 데이터 경로 (README 구조)
RAW_CSV_PATH = "data/raw/urls.csv"             # 원본 URL 데이터
TRAIN_CSV_PATH = "data/processed/train.csv"    # 학습 데이터
VALID_CSV_PATH = "data/processed/valid.csv"    # 검증 데이터
TEST_CSV_PATH = "data/processed/test.csv"      # 평가 데이터

# 전처리 설정
TRAIN_RATIO = 0.8       # 학습 데이터 비율
VALID_RATIO = 0.1       # 검증 데이터 비율
TEST_RATIO = 0.1        # 평가 데이터 비율
SEED = 42               # 재현성을 위한 random seed

# === 혼합 학습셋 / hard example 재학습 ===
HARD_EXAMPLES_PATH = "data/hard_examples/hard_examples.csv"  # OOD 오분류 누적 저장
MIXED_PHIUSIIL_PER_CLASS = 25000  # 혼합 학습셋 PhiUSIIL 클래스별 샘플 수
MIXED_URLHAUS_SAMPLES = 15000     # 혼합 학습셋 URLhaus 악성 샘플 수
MIXED_OPENPHISH_SAMPLES = 5000    # 혼합 학습셋 OpenPhish 악성 샘플 수
MIXED_PHISHTANK_SAMPLES = 10000   # 혼합 학습셋 PhishTank 악성 샘플 수
MIXED_TRANCO_SAMPLES = 40000      # 혼합 학습셋 Tranco 정상 샘플 수
MIXED_TRANCO_TOP_N = 100000       # 혼합 학습용 Tranco 정상 후보 상위 N개
DYNAMIC_AUG_PER_CLASS = 5000      # 동적 URL(query/path/token) synthetic augmentation 수
HARD_EXAMPLES_MAX_PER_CLASS = 2000   # 재학습에 섞을 hard example 클래스별 최대 수

# === Tabular feature 통합 (이슈 #3) ===
# URL 문자열만으로 즉시 계산 가능 → 학습/평가/추론 모두에서 사용
URL_FEATURE_COLS = [
    "IsHTTPS",
    "URLLength",
    "DomainLength",
    "NoOfSubDomain",
    "IsDomainIP",
    "TLDLength",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",          # PhiUSIIL 원본 오타 그대로
    "DegitRatioInURL",
    "SpacialCharRatioInURL",
    "NoOfOtherSpecialCharsInURL",
    "HasObfuscation",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfAtInURL",
    "NoOfDashInURL",
    "NoOfDotInURL",
    "NoOfPercentInURL",
    "PathLength",
    "QueryLength",
    "NoOfPathSegments",
    "NoOfQueryParams",
    "HasFragment",
]
# HTML/페이지 분석이 필요 → 학습/평가만 PhiUSIIL 사전계산값 사용,
# 단일 URL 추론에서는 train mean으로 imputation
# 주의: URLSimilarityIndex는 라벨과 |corr|=0.86으로 cheat-feature이라 제외함
#       (이슈 #3 ablation 결과; 다시 추가하지 말 것)
HTML_FEATURE_COLS = [
    "HasSocialNet",
    "HasCopyrightInfo",
    "HasDescription",
]
# 현재 배포/외부 평가에서는 HTML을 실제 fetch하지 않으므로 URL 기반 feature만 사용한다.
# HTML feature를 다시 쓰려면 predict/evaluate_ood에서도 동일 feature를 계산해야 한다.
FEATURE_COLS = URL_FEATURE_COLS

FEATURE_NORM_PATH = "saved/feature_norm.json"  # 학습 데이터의 mean/std 저장
TABULAR_HIDDEN = 32                            # tabular 분기의 hidden 차원
