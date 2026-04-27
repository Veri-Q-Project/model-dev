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
    THRESHOLD
)

from dataset import URLDataset, load_vocab
from charcnn import CharCNN


def evaluate():
    # 1. vocab 로드
    vocab = load_vocab()

    # 2. test dataset 생성
    test_dataset = URLDataset(
        csv_path="data/processed/test.csv",
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

    # 4. 예측
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)

            logits = model(x)
            scores = torch.sigmoid(logits)

            preds = (scores >= THRESHOLD).int().cpu().tolist()

            y_true.extend(y.int().tolist())
            y_pred.extend(preds)

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


if __name__ == "__main__":
    evaluate()