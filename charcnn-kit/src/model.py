# charCNN 모델 구조 정의
# input: dataset.py에서 만든 url 문자 번호 배열
import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    NUM_FILTERS,
    KERNEL_SIZES,
    DROPOUT,
    TABULAR_HIDDEN,
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


# 이슈 #3: CharCNN + tabular feature 융합 모델 (late fusion).
# CharCNN 본체 구조는 건드리지 않고, 그 안의 sub-module을 재사용해 자체 forward를 구성.
# 출력 직전에 tabular 분기와 concat → 최종 fc.
class HybridCharCNN(nn.Module):
    def __init__(self, vocab_size: int, num_features: int):
        super().__init__()
        # CharCNN을 만들어 sub-module(embedding/convs/dropout)만 재사용.
        # CharCNN.fc는 입력 차원이 다르므로 사용하지 않고 새 fc로 교체.
        base = CharCNN(vocab_size)
        self.embedding = base.embedding   # explain.py가 model.embedding 참조
        self.convs = base.convs
        self.dropout = base.dropout
        char_out_dim = base.fc.in_features  # NUM_FILTERS * len(KERNEL_SIZES)

        # tabular 분기: 표준화된 feature 벡터 → 작은 MLP
        self.tab_branch = nn.Sequential(
            nn.Linear(num_features, TABULAR_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )

        # 최종 fc: char vector + tab vector concat
        self.fc = nn.Linear(char_out_dim + TABULAR_HIDDEN, 1)

    def forward(self, x_url, x_tab):
        e = self.embedding(x_url)
        e = e.permute(0, 2, 1)  # [B, embed, len]

        pooled = []
        for conv in self.convs:
            c = conv(e)
            c = torch.relu(c)
            c = torch.max(c, dim=2)[0]
            pooled.append(c)
        char_vec = torch.cat(pooled, dim=1)
        char_vec = self.dropout(char_vec)

        tab_vec = self.tab_branch(x_tab)

        z = torch.cat([char_vec, tab_vec], dim=1)
        return self.fc(z).squeeze(1)
