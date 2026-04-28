# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트
URL 문자열을 입력 받아 phishing 여부를 분류하는 CharCNN 모델. 본 레포(`model-dev`)는 모델 개발용이고, 실제 코드는 서브디렉터리 `charcnn-kit/`에 있다.

## 작업 디렉터리
**모든 Python 명령은 `charcnn-kit/` 안에서 실행해야 한다.** 코드가 `data/`, `saved/` 같은 상대경로를 사용하기 때문이다.
```bash
cd charcnn-kit
```

## 파이프라인 (순서)
```bash
pip install -r requirements.txt
python src/download_phiusiil.py   # 데이터 다운로드 (UCI PhiUSIIL, 5만 균형 샘플)
python src/preprocess.py          # raw -> train/valid/test (8:1:1 stratified)
python src/train.py               # 학습 (early stopping + best-model 저장)
python src/evaluate.py            # 지표 + FN/FP URL 출력
python src/predict.py --url "..." [--xai]  # 단일 URL 예측
```
대안 데이터 소스: `python src/download_data.py` (URLhaus + Tranco). 두 소스 형식 차이로 모델이 trivially 분리되므로 baseline용으론 부적합하다.

## 절대 어겨선 안 되는 컨벤션

- **라벨 규칙**: `0 = 정상`, `1 = 악성`. README에서도 명시한 고정 규칙. PhiUSIIL은 반대 규칙(1=legit, 0=phish)이라 `download_phiusiil.py`에서 라벨을 반전시킨다 — 이 반전 로직은 깨면 안 됨.
- **수정 금지 영역** (README 8번):
  - `dataset.py`의 문자 인코딩 구조 (`build_vocab`, `encode_url`, PAD/UNK 토큰)
  - `model.py`의 CharCNN 핵심 구조
  - 라벨 기준 자체
- **vocab 생애주기**: vocab은 `train.py`에서 train.csv로부터 생성되어 `saved/char_vocab.json`에 저장된다. evaluate/predict는 **반드시 저장된 vocab을 로드**해야 한다 (새로 만들면 character→index 매핑이 달라져 모델이 깨짐).

## 아키텍처 (큰 그림)

**데이터 흐름:**
```
data/raw/urls.csv (url, label)
  └─ preprocess.py → stratified 8/1/1 split, 라벨 검증, 중복/null 제거
data/processed/{train,valid,test}.csv
  └─ URLDataset (dataset.py) → char-level encode_url(URL) → tensor
torch.DataLoader
  └─ CharCNN (model.py): Embedding → 여러 Conv1d kernel → max-pool over time → concat → Dropout → FC(1) → BCEWithLogitsLoss
saved/charcnn.pt + saved/char_vocab.json
```

**CharCNN 구조 핵심:** 문자별 임베딩을 Conv1d 여러 kernel size(`KERNEL_SIZES`)로 통과시켜 각각의 max-pool 결과를 concat. URL 길이는 `MAX_LEN`(200)으로 패딩/truncate.

**학습 흐름 (`train.py`):**
- valid_loss를 epoch마다 계산
- `valid_loss`가 갱신될 때마다 best state를 `copy.deepcopy`로 보관
- `PATIENCE` epoch 동안 개선 없으면 early stopping
- 최종 저장은 **마지막 epoch이 아닌 best epoch의 state** (이게 핵심)

**Config (`config.py`):** 모든 하이퍼파라미터/경로의 단일 출처. README 9번에 추천 튜닝 순서가 정리되어 있음.

## 알려진 제약

- **CharCNN의 본질적 한계**: URL 문자열만 보므로 "평범해 보이는 신규 phishing 도메인"은 구분 불가. 현재 baseline의 FN 9건이 모두 이 패턴(`https://www.{도메인}.{tld}` 단순 형태). recall을 더 올리려면 WHOIS/DNS feature 통합이 필요 — 모델 구조 튜닝만으로는 한계.
- **Windows 콘솔에서 한글 mojibake**: print의 한글이 깨져 보일 수 있음(예: `악성` → `��`). 코드/저장 파일은 UTF-8로 정상이며, **콘솔 표시 문제일 뿐**이므로 무시해도 된다.
- **재현 가능 산출물은 .gitignore됨**: `data/raw/*.csv`, `data/processed/*.csv`, `saved/*.pt`, `saved/*.json`. 새 환경에서 작업할 땐 위 파이프라인을 처음부터 돌려야 한다.

## 현재 baseline (참고용)
- PhiUSIIL test 5,000건: Accuracy 0.9982, Precision 1.0000, Recall 0.9964, FN 9, FP 0
- README 목표(recall ≥0.90, accuracy ~0.90)는 충족 상태. 다음 작업의 비교 기준점.
