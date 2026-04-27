# 설정값 관리
import torch

MAX_LEN = 200               # URL 최대 길이
BATCH_SIZE = 32             # 배치 크기: 한번에 몇 개 데이터를 묶어 학습할것인가
EPOCHS = 10                 # 에폭 수: 전체 학습 데이터를 몇 번 반복 학습할지 (너무 많을시 과적합 발생 가능)
LEARNING_RATE = 0.001       # 학습률: 한 번 업데이트 시 얼마나 수정을 거칠지
EMBED_DIM = 64              # 문자 임베딩 차원 수: 각 문자를 몇 개 숫자로 표현할지
NUM_FILTERS = 128           # CNN 필터 수 = url 패턴 찾는 탐지기 수 (너무 많을시 과적합 발생 가능)
KERNEL_SIZES = [3, 4, 5]    # 패턴 크기 목록(필터): 3글자(예: log), 4글자(예: bank), 5글자(예: login)
DROPOUT = 0.3               # 드롭아웃 비율: 학습 중 몇%의 뉴련을 쉬게 할지 (과적합 방지)

# GPU 있으면 GPU 사용-> 없으면 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pridict
THRESHOLD = 0.5             # score가 0.5 이상이면 악성으로 판단

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