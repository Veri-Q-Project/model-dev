# 모델 학습 및 학습 결과와 완료된 모델을 저장함
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    EPOCHS,
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

    # 8. 학습 루프
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

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"train_loss: {avg_loss:.4f} | valid_loss: {avg_valid_loss:.4f}"
        )

    # 9. 모델 저장
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vocab saved to {VOCAB_PATH}")


if __name__ == "__main__":
    train()
