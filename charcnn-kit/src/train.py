# 모델 학습 및 학습 결과와 완료된 모델을 저장함
import copy
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    DEVICE,
    MODEL_PATH,
    VOCAB_PATH,
    TRAIN_CSV_PATH,
    VALID_CSV_PATH
)

from dataset import URLDataset, save_vocab
from model import CharCNN


def train():
    os.makedirs("saved", exist_ok=True)

    # 1. 학습 데이터셋 생성
    train_dataset = URLDataset(csv_path=TRAIN_CSV_PATH, build_new_vocab=True)

    # 2. train에서 만든 vocab 저장
    vocab = train_dataset.vocab
    save_vocab(vocab)

    # 3. 검증 데이터셋 생성 (train의 vocab 재사용)
    valid_dataset = URLDataset(csv_path=VALID_CSV_PATH, vocab=vocab)

    # 4. DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 5. 모델 생성
    model = CharCNN(vocab_size=len(vocab)).to(DEVICE)

    # 6. 손실 함수
    criterion = nn.BCEWithLogitsLoss()

    # 7. optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # 8. 학습 루프 (early stopping + best model 추적)
    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # 이전 gradient 초기화
            optimizer.zero_grad()

            # 모델 예측
            logits = model(x)

            # loss 계산
            loss = criterion(logits, y)

            # 역전파
            loss.backward()

            # 파라미터 업데이트
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 검증 loss 계산
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x)
                valid_loss += criterion(logits, y).item()
        avg_valid_loss = valid_loss / max(len(valid_loader), 1)

        # best 모델 갱신 / patience 체크
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
            marker = "  *best"
        else:
            patience_counter += 1
            marker = f"  (no improve {patience_counter}/{PATIENCE})"

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"train_loss: {avg_loss:.4f} | valid_loss: {avg_valid_loss:.4f}"
            f"{marker}"
        )

        if patience_counter >= PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch + 1} "
                f"(best epoch: {best_epoch}, best valid_loss: {best_valid_loss:.4f})"
            )
            break

    # 9. best 모델 복원 후 저장
    if best_state is None:
        raise RuntimeError("학습된 모델 state가 없습니다.")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    print(
        f"Saved best model (epoch {best_epoch}, "
        f"valid_loss {best_valid_loss:.4f}) to {MODEL_PATH}"
    )
    print(f"Vocab saved to {VOCAB_PATH}")


if __name__ == "__main__":
    train()
