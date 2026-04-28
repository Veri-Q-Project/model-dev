# 모델의 성능을 측정합니다.
# accuracy, precision, recall, f1
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from config import (
    DEVICE,
    MODEL_PATH,
    BATCH_SIZE,
    THRESHOLD,
    TEST_CSV_PATH
)

from dataset import URLDataset, load_vocab
from model import CharCNN


def evaluate():
    # 1. vocab 로드
    vocab = load_vocab()

    # 2. test dataset 생성
    test_dataset = URLDataset(
        csv_path=TEST_CSV_PATH,
        vocab=vocab
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 3. 모델 로드
    model = CharCNN(vocab_size=len(vocab)).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    y_true = []
    y_pred = []
    y_score = []

    # 4. 예측
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)

            logits = model(x)
            scores = torch.sigmoid(logits)

            preds = (scores >= THRESHOLD).int().cpu().tolist()

            y_true.extend(y.int().tolist())
            y_pred.extend(preds)
            y_score.extend(scores.cpu().tolist())

    # 5. 지표 계산
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("=== Evaluation Result ===")
    print(f"Accuracy : {acc:.4f}")      # 전체 중 맞춘 개수
    print(f"Precision: {prec:.4f}")     # 악성으로 분류한 것 중 실제 악성 비율
    print(f"Recall   : {rec:.4f}")      # 실제 악성 url을 악성으로 분류한 개수
    print(f"F1 Score : {f1:.4f}")       # precision+recall 균형 점수
    print("Confusion Matrix:")
    print(cm)

    # 6. 오분류 사례 출력 (FN: 악성을 정상으로, FP: 정상을 악성으로)
    urls = test_dataset.df["url"].tolist()
    fn_cases = []   # 미탐 (실제 1, 예측 0)
    fp_cases = []   # 오탐 (실제 0, 예측 1)
    for url, t, p, s in zip(urls, y_true, y_pred, y_score):
        if t == 1 and p == 0:
            fn_cases.append((s, url))
        elif t == 0 and p == 1:
            fp_cases.append((s, url))

    print(f"\n=== False Negatives (미탐: 악성을 정상으로 판단) [{len(fn_cases)}건] ===")
    # 점수 높은 순(0.5에 가까웠던 것부터)으로 출력
    for s, url in sorted(fn_cases, key=lambda x: -x[0]):
        print(f"  score={s:.4f}  {url}")

    print(f"\n=== False Positives (오탐: 정상을 악성으로 판단) [{len(fp_cases)}건] ===")
    for s, url in sorted(fp_cases, key=lambda x: -x[0]):
        print(f"  score={s:.4f}  {url}")


if __name__ == "__main__":
    evaluate()
