# charCNN 모델 구조 정의
# input: dataset.py에서 만든 url 문자 번호 배열
import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    NUM_FILTERS,
    KERNEL_SIZES,
    DROPOUT
)


class CharCNN(nn.Module):
    def __init__(self, vocab_size):
        # vocab_size: 문자 사전의 크기(각 항목의 총 개수)
        super().__init__()

        # 문자 index를 벡터로 변환
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIM,
            padding_idx=0
        )

        # 여러 크기의 CNN 필터 생성
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=EMBED_DIM,
                out_channels=NUM_FILTERS,
                kernel_size=k
            )
            for k in KERNEL_SIZES
        ])

        # 과적합 방지
        self.dropout = nn.Dropout(DROPOUT)

        # 최종 이진 분류 출력층
        self.fc = nn.Linear(
            NUM_FILTERS * len(KERNEL_SIZES),
            1
        )

    # x shape: [batch_size, MAX_LEN]
    def forward(self, x):
        # [batch, max_len]
        x = self.embedding(x)   # 숫자 하나를 유의미한 좌표 벡터로 변환

        # [batch, max_len, embed_dim]
        # Conv1d는 channel이 가운데 와야 함
        x = x.permute(0, 2, 1)

        # [batch, embed_dim, max_len]

        conv_results = []

        for conv in self.convs:
            c = conv(x)              # CNN 적용
            c = torch.relu(c)           # 활성화 함수: 쓸모없는 음수 제거, 특징 강조
            c = torch.max(c, dim=2)[0]  # url전체에서 가장 강하게 감지된 패턴만 가져옴
            conv_results.append(c)

        # 각 필터 결과 합치기
        x = torch.cat(conv_results, dim=1)

        # dropout: 과적합 방지
        x = self.dropout(x)

        # 최종 점수(logit): 양수가 크면 악성, 음수가 크면 정상.
        x = self.fc(x)
        # 다른 곳에서 sigmoid를 씌우면 확률이 됩니다.

        # 각 url의 점수를 출력
        return x.squeeze(1)