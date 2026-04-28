# CharCNN URL 악성 탐지 모델 안내문

## 1. 프로젝트 개요

CharCNN 모델의 기본 뼈대입니다.
**진행 목표 사항**은
1.`데이터셋 확장`
2.`성능 향상(recall 90% 이상, accuracy 90% 내외, FN최소화)`
3.`모델 구조 개선(필터 수 증가, kernel size 추가 등...)`
이 있습니다.

**역할 분담**: **7**번 항목의 내용을 참고해서 자유롭게 해 주시면 됩니다. 학습+평가/데이터로 분할하는 것을 추천드리는 편입니다.

입력:

```text
http://paypa1-login.xyz
```

출력:

```json
{
  "score": 0.91,
  "label": "malicious"
}
```

의미:

* `score`: 악성일 확률(0 ~ 1)
* `label`: 최종 판단 결과

---

## 2. 파일 설명

### data/

데이터 파일 저장 폴더입니다.

#### data/raw/

원본 CSV 파일 위치

예:

```csv
url,label
https://google.com,0
http://paypa1-login.xyz,1
```

#### data/processed/

전처리 후 학습용 데이터 저장 위치

* `train.csv`
* `valid.csv`
* `test.csv`

#### data/ood/

외부 출처 OOD 평가 데이터 위치

* `ood_test.csv`

#### data/hard_examples/

OOD 평가에서 틀린 FN/FP URL을 누적 저장하는 위치

* `hard_examples.csv`

---

### saved/

학습 완료 후 생성되는 결과물입니다.

#### charcnn.pt

학습된 모델 파일

#### char_vocab.json

문자를 숫자로 바꾸는 사전 파일

#### feature_norm.json

tabular feature의 train mean/std 저장 파일

---

### src/

실제 코드 폴더입니다.

| 파일명                  | 역할                                |
| -------------------- | --------------------------------- |
| config.py            | 설정값 관리                            |
| preprocess.py        | 데이터 정리 및 train/valid/test stratified 분할 |
| dataset.py           | URL → 숫자 배열 변환                    |
| model.py             | CharCNN 모델 구조                     |
| train.py             | 모델 학습 (early stopping + best-model 저장) |
| evaluate.py          | 성능 평가 + FN/FP 사례 출력               |
| evaluate_ood.py      | 외부 출처 OOD 평가 + threshold sweep      |
| predict.py           | URL 단일 예측                         |
| explain.py           | XAI 로그 기능                         |
| features.py          | URL feature 계산 + feature 정규화 저장/로드 |
| download_phiusiil.py | UCI PhiUSIIL 데이터셋 다운로드 (추천)       |
| download_data.py     | URLhaus + Tranco 데이터 다운로드 (대안)    |
| download_train_mixed.py | PhiUSIIL + URLhaus + OpenPhish + PhishTank + Tranco 혼합 학습셋 생성 |
| download_ood.py      | URLhaus + PhishTank + OpenPhish + Tranco OOD 평가셋 생성 |

---

## 3. 실행 순서

### 1단계. 라이브러리 설치

```bash
pip install -r requirements.txt
```

---

### 2단계. 데이터 준비

`data/raw/urls.csv` 파일이 필요합니다.

형식:

```csv
url,label
https://google.com,0
http://paypa1-login.xyz,1
```

라벨 기준:

* `0` = 정상
* `1` = 악성

#### 옵션 A. 혼합 학습셋 자동 다운로드 (추천)

PhiUSIIL, URLhaus, OpenPhish, PhishTank, Tranco를 섞어 학습 데이터를 만듭니다.
현재 기본 학습 파이프라인은 이 방식을 권장합니다.

```bash
python src/download_train_mixed.py
```

기본 구성:

* 악성 약 50,000개: PhiUSIIL phishing 25,000 + URLhaus recent 15,000 + PhishTank 10,000 + OpenPhish available feed
* 정상 약 50,000개: PhiUSIIL legitimate + Tranco top 100,000 중 샘플
* 동적 URL augmentation 10,000개: 정상/악성 각각 5,000개 query/path/token 변형
* 실행 결과: `data/raw/urls.csv`
* PhishTank app key가 있으면 환경변수 `PHISHTANK_APP_KEY`에 넣으면 됩니다. 없으면 public feed를 시도합니다.

#### 옵션 B. PhiUSIIL만 사용

UCI ML PhiUSIIL Phishing URL Dataset (235k URL)에서 5만개 균형 샘플링:

```bash
python src/download_phiusiil.py
```

실행 결과: `data/raw/urls.csv` (악성 25,000 / 정상 25,000)

> 다른 소스도 가능: `python src/download_data.py` (URLhaus + Tranco). 다만 두 소스 형식 차이로 모델이 trivially 분리되어 baseline용으론 비추천.

#### 옵션 C. 직접 작성

CSV 형식에 맞춰 `data/raw/urls.csv`를 수동으로 채워도 됩니다.

#### 추가로 찾아오면 좋은 데이터 출처

* 악성 URL: URLhaus `https://urlhaus.abuse.ch/`, PhishTank `https://www.phishtank.org/developer_info.php`, OpenPhish `https://openphish.com/phishing_feeds.html`
* 정상 URL: Tranco `https://tranco-list.eu/`, 회사/학교/정부/뉴스/쇼핑/포털 등 실제 정상 사이트 목록
* 직접 가져온 CSV는 최소 `url,label` 컬럼을 맞추면 됩니다. 라벨은 `0=정상`, `1=악성`입니다.
* 정상 후보는 악성 feed와 겹치는 URL을 제거해야 합니다.

---

### 3단계. 전처리

```bash
python src/preprocess.py
```

결과:

```text
data/processed/train.csv
data/processed/valid.csv
data/processed/test.csv
```

기본 전처리는 같은 등록 도메인이 train/valid/test에 동시에 들어가지 않도록 domain group split을 사용합니다. 이 설정은 과적합과 데이터 누수를 줄이기 위한 것입니다.
URL feature는 query/path/token 같은 동적 URL 대응을 위해 `NoOfQueryParams`, `QueryLength`, `PathLength`, `NoOfPathSegments`, `HasFragment` 등을 포함합니다.

---

### 4단계. 학습

```bash
python src/train.py
```

결과:

```text
saved/charcnn.pt
saved/char_vocab.json
saved/feature_norm.json
```

예상 로그:

```text
Epoch [1/20] train_loss: 0.0288 | valid_loss: 0.0176  *best
Epoch [2/20] train_loss: 0.0162 | valid_loss: 0.0173  *best
Epoch [3/20] train_loss: 0.0129 | valid_loss: 0.0159  *best
...
Early stopping triggered at epoch 8 (best epoch: 5, ...)
Saved best model (epoch 5, valid_loss 0.0142) to saved/charcnn.pt
```

* train_loss가 감소하면 정상입니다.
* valid_loss가 PATIENCE epoch 동안 개선이 없으면 early stopping이 종료시킵니다.
* 종료 시점에 저장되는 모델은 **valid_loss가 가장 낮았던 시점의 best 모델**입니다 (마지막 epoch이 아님).

---

### 5단계. 평가

```bash
python src/evaluate.py
```

출력 예:

```text
=== Evaluation Result ===
Accuracy : 0.9975
Precision: 0.9980
Recall   : 0.9969
F1 Score : 0.9975
Confusion Matrix:
[[5509   11]
 [  17 5504]]

=== False Negatives (미탐: 악성을 정상으로 판단) [17건] ===
  score=0.1295  https://verificahype.simply.site
  ...

=== False Positives (오탐: 정상을 악성으로 판단) [11건] ===
```

* 지표 외에 **FN/FP 사례가 score와 함께 출력**되어 모델이 어떤 URL을 놓쳤는지 분석 가능합니다.

---

### 6단계. 외부 출처 OOD 평가 + hard example 저장

PhiUSIIL 내부 split 성능은 실제 인터넷 URL 성능을 보장하지 않습니다.
외부 데이터로 별도 평가합니다.

```bash
python src/download_ood.py
python src/evaluate_ood.py
```

기본 설정:

* 악성: URLhaus recent 5,000개
* 정상: Tranco 상위 100,000 도메인 중 5,000개 샘플
* 저장 위치: `data/ood/ood_test.csv`
* `download_ood.py`는 기본적으로 `data/raw/urls.csv`에 들어간 학습 URL을 제외하고 OOD 평가셋을 만듭니다.
* 현재 모델은 실제 추론에서 계산 가능한 URL 기반 feature만 사용합니다.
* `evaluate_ood.py`는 기본적으로 오분류 FN/FP를 `data/hard_examples/hard_examples.csv`에 누적 저장합니다.

hard example을 다음 학습에 반영하는 루프:

```bash
python src/evaluate_ood.py --save-hard
python src/download_train_mixed.py
python src/preprocess.py
python src/train.py
python src/download_ood.py
python src/evaluate_ood.py
```

과적합 방지:

* hard example은 클래스별 최대 `HARD_EXAMPLES_MAX_PER_CLASS`개만 학습에 섞습니다.
* hard example의 등록 도메인 group은 train에 고정하고 valid/test와 겹치지 않게 합니다.
* 같은 OOD 파일에 대해 무한 반복하지 말고, 최신 feed로 새 OOD를 만든 뒤 반복합니다.

2026-04-28 현재 HybridCharCNN OOD 결과:

```text
Threshold 0.20
Accuracy : 0.9975
Precision: 0.9988
Recall   : 0.9962
F1 Score : 0.9975
Confusion Matrix [[TN FP], [FN TP]]:
[[ 4994     6]
 [   19  4981]]
```

해석:

* 이전 PhiUSIIL+HTML-feature 모델은 Tranco 정상 5,000개 중 4,981개를 오탐했습니다.
* 혼합 학습셋 + URL 기반 feature-only 모델로 바꾼 뒤 대량 오탐 문제는 해소됐습니다.
* 현재 OOD 악성은 URLhaus + PhishTank 잔여 URL을 포함합니다. OOD FN은 PhishTank 쪽에서 발생합니다.
* 동적 URL augmentation 후 OOD는 이전 FN 14 / FP 23에서 FN 12 / FP 15로 개선됐습니다.
* hard example 1회 재학습 후 OOD 오탐은 15개에서 6개로 줄었고, 내부 test도 FP 17개에서 11개로 줄었습니다. 대신 OOD FN은 12개에서 19개로 늘어 threshold/정책 선택이 필요합니다.
* OpenPhish community feed는 현재 샘플 수가 작아 추가 검증용으로는 부족합니다. 직접 수집 정상 URL과 최신 phishing feed를 계속 추가해야 합니다.

---

### 7단계. 단일 URL 예측

```bash
python src/predict.py --url "http://paypa1-login.xyz"
```

출력:

```json
{
  "score": 0.91,
  "label": "malicious"
}
```

#### Xai 설명 버전

```bash
python src/predict.py --url "http://paypa1-login.xyz" --xai
```

출력:

```json
{
  "score": 0.91,
  "label": "malicious",
  "xai": {
    "top_chars": []
  }
}
```

---

## 4. config.py 수정 가능 항목

```python
MAX_LEN
BATCH_SIZE
EPOCHS         # 최대 에폭 (early stopping이 실제 종료 시점 결정)
PATIENCE       # early stopping: valid_loss가 N epoch 동안 개선 없으면 중단
LEARNING_RATE
EMBED_DIM
NUM_FILTERS
KERNEL_SIZES
DROPOUT
THRESHOLD
```

---

## 5. 성능 지표 설명

| 지표        | 의미                    |
| --------- | --------------------- |
| Accuracy  | 전체 정확도                |
| Precision | 악성이라 한 것 중 진짜 악성 비율   |
| Recall    | 실제 악성 중 잡아낸 비율        |
| F1 Score  | Precision + Recall 균형 |

### 중요

보안 탐지에서는 **Recall**이 매우 중요합니다.

악성 URL을 놓치면 위험합니다.

---

## 6. 자주 발생하는 오류

### 모델 파일 없음

```text
saved/charcnn.pt 없음
```

해결:

```bash
python src/train.py
```

---

### vocab 없음

```text
saved/char_vocab.json 없음
```

해결:

```bash
python src/train.py
```

---

### torch 없음

```text
ModuleNotFoundError: torch
```

해결:

```bash
pip install -r requirements.txt
```

---

### 성능이 이상함

예:

```text
모든 URL이 정상
모든 URL이 악성
accuracy 낮음
```

확인 사항:

* 데이터 라벨 오류
* 정상/악성 비율 불균형
* 데이터 수 부족
* learning rate 문제

---

## 7. 역할 분담 추천

### 데이터 담당

* URL 수집
* 라벨 검수
* 중복 제거
* 데이터 균형 조절

### 학습 담당

* train.py 실행
* config 튜닝
* 최고 성능 모델 기록

### 평가 담당

* evaluate.py 실행
* confusion matrix 분석
* 오탐/미탐 URL 정리

---

## 8. 수정 주의사항

### 수정 가능

* config.py 값

### 함부로 수정 금지

* dataset.py 문자 인코딩 구조
* model.py 핵심 구조
* explain.py 내부 로직 **단순 설명 보조용이므로 방해되면 삭제해도 OK
* label 기준 (`0=정상`, `1=악성`)

---

## 9. 추천 튜닝 순서

### 1차

```text
EPOCHS 증가
10 → 20 → 30
```

### 2차

```text
LEARNING_RATE 변경
0.001 / 0.0005 / 0.0001
```

### 3차

```text
BATCH_SIZE 조절
16 / 32 / 64
```

### 4차

```text
MAX_LEN 조절
128 / 200 / 256
```

### 5차

```text
DROPOUT 조절
0.3 / 0.5
```
