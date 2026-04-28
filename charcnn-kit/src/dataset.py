# url 문자열을 문자 단위 숫자 배열로 변환
# (이슈 #3) feature_cols가 주어지면 tabular feature도 함께 반환
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import MAX_LEN, VOCAB_PATH


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# csv 읽기 -> 문자 사전 생성
def build_vocab(urls):
    vocab = {
        PAD_TOKEN: 0,  # 길이를 맞출 때 채우는 값
        UNK_TOKEN: 1,  # 사전에 없는 문자가 나왔을 때 쓰는 값
    }

    for url in urls:
        for ch in str(url):
            if ch not in vocab:
                vocab[ch] = len(vocab)

    return vocab


# 문자 사전을 JSON파일로 저장
def save_vocab(vocab, path=VOCAB_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


# 저장된 문자 사전 불러오기
def load_vocab(path=VOCAB_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# url 문자열 -> 숫자 index로 변환
# 길이 < MAX_LEN : 0으로 패딩
# 길면 MAX_LEN 길이까지만 자름
def encode_url(url, vocab, max_len=MAX_LEN):
    url = str(url)

    encoded = []
    for ch in url:
        encoded.append(vocab.get(ch, vocab[UNK_TOKEN]))

    if len(encoded) < max_len:
        encoded += [vocab[PAD_TOKEN]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    return encoded


# pytorch dataloader가 읽을 수 있는 url데이터셋 클래스
# 기본: (url 숫자 배열, label)
# feature_cols 지정 시: (url 숫자 배열, tabular feature 벡터, label)
class URLDataset(Dataset):
    def __init__(
        self,
        csv_path,
        vocab=None,
        build_new_vocab=False,
        feature_cols=None,
    ):
        self.df = pd.read_csv(csv_path)

        self.df = self.df.dropna(subset=["url", "label"])
        self.df["url"] = self.df["url"].astype(str)
        self.df["label"] = self.df["label"].astype(int)

        if build_new_vocab:
            self.vocab = build_vocab(self.df["url"].tolist())
        else:
            if vocab is None:
                raise ValueError("vocab이 필요합니다.")
            self.vocab = vocab

        self.feature_cols = list(feature_cols) if feature_cols else None
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in self.df.columns]
            if missing:
                raise ValueError(f"CSV에 feature 컬럼이 없습니다: {missing}")
            # 결측 row를 추가로 떨어뜨림 (preprocess에서 이미 처리되지만 안전망)
            self.df = self.df.dropna(subset=self.feature_cols).reset_index(drop=True)
            self.features = (
                self.df[self.feature_cols].astype("float32").to_numpy()
            )
        else:
            self.features = None

    def apply_normalization(self, mean: np.ndarray, std: np.ndarray):
        """학습 데이터로 fit한 mean/std로 features를 표준화 (in-place).

        표준화: (x - mean) / std → 각 feature를 평균 0, 표준편차 1로 맞춰
        스케일이 다른 컬럼들이 학습을 망가뜨리지 않게 함.
        """
        if self.features is None:
            raise RuntimeError("feature_cols 없이 정규화는 불가합니다.")
        std_safe = np.where(std == 0, 1.0, std)
        self.features = ((self.features - mean) / std_safe).astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        url = self.df.iloc[idx]["url"]
        label = self.df.iloc[idx]["label"]

        x = encode_url(url, self.vocab)
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.float32)

        if self.features is None:
            return x, y

        f = torch.from_numpy(self.features[idx])
        return x, f, y
