# 모델 학습 및 학습 결과와 완료된 모델을 저장함
# (이슈 #3) HybridCharCNN: CharCNN + URL 기반 tabular feature 융합
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
    VALID_CSV_PATH,
    FEATURE_COLS,
)

from dataset import URLDataset, save_vocab
from model import HybridCharCNN
from features import save_norm


def train():
    os.makedirs("saved", exist_ok=True)

    # 1. 학습 데이터셋 (vocab + tabular feature 모두 빌드)
    train_dataset = URLDataset(
        csv_path=TRAIN_CSV_PATH,
        build_new_vocab=True,
        feature_cols=FEATURE_COLS,
    )
    vocab = train_dataset.vocab
    save_vocab(vocab)

    # 2. tabular feature 정규화 (mean/std 계산은 train 데이터에서만 — 데이터 누수 방지)
    #    표준화: (x - mean) / std → 평균 0, 표준편차 1로 맞춤. 스케일이 다른 feature들이
    #    학습을 한쪽으로 끌어당기지 않게 하는 표준 전처리.
    mean = train_dataset.features.mean(axis=0)
    std = train_dataset.features.std(axis=0)
    train_dataset.apply_normalization(mean, std)
    save_norm(mean, std)
    print(f"[norm] {len(FEATURE_COLS)}개 feature mean/std 저장됨")

    # 3. 검증 데이터셋 (train의 vocab + 동일 mean/std 사용)
    valid_dataset = URLDataset(
        csv_path=VALID_CSV_PATH,
        vocab=vocab,
        feature_cols=FEATURE_COLS,
    )
    valid_dataset.apply_normalization(mean, std)

    # 4. DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 모델
    model = HybridCharCNN(
        vocab_size=len(vocab),
        num_features=len(FEATURE_COLS),
    ).to(DEVICE)

    # 6. 손실 함수 / optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. 학습 루프 (early stopping + best model 추적)
    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for x, f, y in train_loader:
            x = x.to(DEVICE)
            f = f.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x, f)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 검증 loss
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x, f, y in valid_loader:
                x = x.to(DEVICE)
                f = f.to(DEVICE)
                y = y.to(DEVICE)
                logits = model(x, f)
                valid_loss += criterion(logits, y).item()
        avg_valid_loss = valid_loss / max(len(valid_loader), 1)

        # best 갱신 / patience
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

    # 8. best 모델 복원 후 저장
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
