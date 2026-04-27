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
    VOCAB_PATH
)

from dataset import URLDataset, save_vocab
from charcnn import CharCNN


def train():
    os.makedirs("saved", exist_ok=True)

    # 1. 학습 데이터셋 생성
    train_dataset = URLDataset(build_new_vocab=True)

    # 2. train에서 만든 vocab 저장
    vocab = train_dataset.vocab
    save_vocab(vocab)

    # 3. DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 4. 모델 생성
    model = CharCNN(vocab_size=len(vocab)).to(DEVICE)

    # 5. 손실 함수
    criterion = nn.BCEWithLogitsLoss()

    # 6. optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # 7. 학습 루프
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

        print(f"Epoch [{epoch + 1}/{EPOCHS}] train_loss: {avg_loss:.4f}")

    # 8. 모델 저장
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vocab saved to {VOCAB_PATH}")


if __name__ == "__main__":
    train()