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
python src/download_train_mixed.py # 데이터 다운로드 (PhiUSIIL + URLhaus + OpenPhish + PhishTank + Tranco 혼합)
python src/preprocess.py          # raw -> train/valid/test (domain-group 8:1:1 split)
python src/train.py               # 학습 (early stopping + best-model 저장)
python src/evaluate.py            # 지표 + FN/FP URL 출력
python src/download_ood.py        # 학습 URL 제외 OOD 평가셋 생성
python src/evaluate_ood.py        # OOD 지표 + threshold sweep + hard example 저장
python src/predict.py --url "..." [--xai]  # 단일 URL 예측
```
대안 데이터 소스: `python src/download_phiusiil.py` (UCI PhiUSIIL만 사용), `python src/download_data.py` (URLhaus + Tranco만 사용). 단일 출처/두 출처만 쓰면 source bias가 생기기 쉬우므로 현재 기본은 혼합 학습셋이다. PhishTank app key가 있으면 `PHISHTANK_APP_KEY` 환경변수 사용.

## 절대 어겨선 안 되는 컨벤션

- **라벨 규칙**: `0 = 정상`, `1 = 악성`. README에서도 명시한 고정 규칙. PhiUSIIL은 반대 규칙(1=legit, 0=phish)이라 `download_phiusiil.py`에서 라벨을 반전시킨다 — 이 반전 로직은 깨면 안 됨.
- **수정 금지 영역** (README 8번):
  - `dataset.py`의 문자 인코딩 구조 (`build_vocab`, `encode_url`, PAD/UNK 토큰)
  - `model.py`의 `CharCNN` 핵심 구조 (CharCNN 클래스는 그대로 두고 새 클래스 `HybridCharCNN`이 sub-module만 재사용)
  - 라벨 기준 자체
- **vocab 생애주기**: vocab은 `train.py`에서 train.csv로부터 생성되어 `saved/char_vocab.json`에 저장된다. evaluate/predict는 **반드시 저장된 vocab을 로드**해야 한다 (새로 만들면 character→index 매핑이 달라져 모델이 깨짐).
- **feature 정규화 통계 생애주기**: tabular feature mean/std는 `train.py`에서 train.csv로만 fit하여 `saved/feature_norm.json`에 저장. valid/test/predict는 반드시 저장된 통계를 로드해 적용 — train 외 데이터의 통계를 섞으면 data leakage가 됨.
- **금지 feature**: `URLSimilarityIndex`는 PhiUSIIL에서 라벨과 |corr|=0.86으로 cheat feature 역할. `FEATURE_COLS`에 절대 추가하지 말 것 (이슈 #3 ablation에서 확인).

## 아키텍처 (큰 그림)

**데이터 흐름 (이슈 #3 이후):**
```
data/raw/urls.csv (url, label, source)
  └─ preprocess.py → domain-group 8/1/1 split, 라벨 검증, 중복/feature-NaN 제거
data/processed/{train,valid,test}.csv
  └─ URLDataset (dataset.py) → (char-encoded URL, tabular feature 벡터, label)
                              → train의 mean/std로 표준화
torch.DataLoader
  └─ HybridCharCNN (model.py):
       ├─ char 분기: Embedding → 여러 Conv1d kernel → max-pool over time → concat → Dropout
       ├─ tabular 분기: Linear → ReLU → Dropout (작은 MLP)
       └─ 두 분기 concat → FC(1) → BCEWithLogitsLoss
saved/charcnn.pt + saved/char_vocab.json + saved/feature_norm.json
```

**CharCNN 구조 핵심:** 문자별 임베딩을 Conv1d 여러 kernel size(`KERNEL_SIZES`)로 통과시켜 각각의 max-pool 결과를 concat. URL 길이는 `MAX_LEN`(200)으로 패딩/truncate. CharCNN 클래스 자체는 보존되고, `HybridCharCNN`이 그 sub-module(embedding/convs/dropout)을 재사용해 forward를 새로 구성한다.

**Tabular feature (이슈 #3):**
- `FEATURE_COLS = URL_FEATURE_COLS` (25개). URL 문자열에서 즉시 계산 가능한 feature만 학습/평가/추론 전부에서 동일하게 사용한다.
- 동적 URL 대응 feature 포함: query/path/token 관련 `NoOfQueryParams`, `QueryLength`, `PathLength`, `NoOfPathSegments`, `HasFragment`, `NoOfEqualsInURL`, `NoOfAmpersandInURL` 등.
- `HTML_FEATURE_COLS`는 현재 비활성화. 실제 predict/evaluate_ood에서 페이지 fetch 없이 mean imputation만 하게 되면 OOD 오탐이 커지므로 다시 추가하지 말 것.

**학습 흐름 (`train.py`):**
- valid_loss를 epoch마다 계산
- `valid_loss`가 갱신될 때마다 best state를 `copy.deepcopy`로 보관
- `PATIENCE` epoch 동안 개선 없으면 early stopping
- 최종 저장은 **마지막 epoch이 아닌 best epoch의 state** (이게 핵심)
- tabular feature의 mean/std도 train에서만 fit해 `saved/feature_norm.json`에 저장

**Config (`config.py`):** 모든 하이퍼파라미터/경로/feature 컬럼 목록의 단일 출처. README 9번에 추천 튜닝 순서가 정리되어 있음.
**Split/Augmentation/Hard mining:** `USE_DOMAIN_GROUP_SPLIT=True`가 기본. 같은 등록 도메인이 train/valid/test에 섞이지 않도록 `StratifiedGroupKFold`를 사용한다. random row split으로 되돌리면 성능은 올라가 보여도 데이터 누수 가능성이 커진다. `download_train_mixed.py`는 정상/악성 각각 5,000개의 동적 URL(query/path/token) synthetic augmentation을 추가한다. `evaluate_ood.py`는 FN/FP를 `data/hard_examples/hard_examples.csv`에 저장하고, 다음 `download_train_mixed.py` 실행 때 클래스별 최대 `HARD_EXAMPLES_MAX_PER_CLASS`개만 재학습에 섞는다. hard example 도메인 group은 train에 고정한다.

## 알려진 제약

- **CharCNN/URL-feature의 한계**: URL 문자열만 보면 "평범해 보이는 신규 phishing 도메인"은 놓칠 수 있다. 그래서 URLhaus/Tranco/PhishTank/OpenPhish 등 외부 출처 OOD 평가를 계속 돌려야 한다.
- **PhiUSIIL HTML feature의 OOD 위험**: `URLSimilarityIndex`는 라벨과 사실상 1:1 → 학습용 `FEATURE_COLS`에서 제외. `HasSocialNet`, `HasCopyrightInfo`, `HasDescription`도 실제 추론에서 계산하지 않으면 train/test 분포가 어긋난다. 현재는 HTML feature 비활성화 상태.
- **Windows 콘솔에서 한글 mojibake**: print의 한글이 깨져 보일 수 있음(예: `악성` → `��`). 코드/저장 파일은 UTF-8로 정상이며, **콘솔 표시 문제일 뿐**이므로 무시해도 된다.
- **재현 가능 산출물은 .gitignore됨**: `data/raw/*.csv`, `data/processed/*.csv`, `saved/*.pt`, `saved/*.json`. 새 환경에서 작업할 땐 위 파이프라인을 처음부터 돌려야 한다.

## 현재 baseline (참고용)
- **Mixed + URL-feature HybridCharCNN** (2026-04-28, domain split, dynamic URL augmentation, hard-mining 1회, PhiUSIIL+URLhaus+OpenPhish+PhishTank+Tranco): internal test Accuracy 0.9975, Precision 0.9980, Recall 0.9969, FN 17, FP 11 at threshold 0.2.
- **OOD(URLhaus+PhishTank/Tranco, 학습 URL 제외)**: Accuracy 0.9975, Precision 0.9988, Recall 0.9962, FN 19, FP 6 at threshold 0.2. Hard mining은 오탐 안정성을 개선했지만 FN은 증가했으므로 recall 우선 운영이면 threshold 0.1 후보도 함께 비교할 것.
- README 목표(recall ≥0.90, accuracy ~0.90)는 둘 다 충족.
