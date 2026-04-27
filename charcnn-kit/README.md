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

---

### saved/

학습 완료 후 생성되는 결과물입니다.

#### charcnn.pt

학습된 모델 파일

#### char_vocab.json

문자를 숫자로 바꾸는 사전 파일

---

### src/

실제 코드 폴더입니다.

| 파일명           | 역할             |
| ------------- | -------------- |
| config.py     | 설정값 관리         |
| preprocess.py | 데이터 정리 및 분할    |
| dataset.py    | URL → 숫자 배열 변환 |
| model.py      | CharCNN 모델 구조  |
| train.py      | 모델 학습          |
| evaluate.py   | 성능 평가          |
| predict.py    | URL 단일 예측      |
| explain.py    | XAI 로그 기능      |

---

## 3. 실행 순서

### 1단계. 라이브러리 설치

```bash
pip install -r requirements.txt
```

---

### 2단계. 데이터 준비

`data/raw/urls.csv`

형식:

```csv
url,label
https://google.com,0
http://paypa1-login.xyz,1
```

라벨 기준:

* `0` = 정상
* `1` = 악성

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

---

### 4단계. 학습

```bash
python src/train.py
```

결과:

```text
saved/charcnn.pt
saved/char_vocab.json
```

예상 로그:

```text
Epoch 1 train_loss: 0.68
Epoch 2 train_loss: 0.55
Epoch 3 train_loss: 0.42
```

loss가 감소하면 정상입니다.

---

### 5단계. 평가

```bash
python src/evaluate.py
```

출력 예:

```text
Accuracy : 0.91
Precision: 0.88
Recall   : 0.94
F1 Score : 0.91
```

---

### 6단계. 단일 URL 예측

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
EPOCHS
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
