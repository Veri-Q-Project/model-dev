# url 문자열을 문자 단위 숫자 배열로 변환
import json
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
# 각 데이터는 (url 숫자 배열, label) 형으로 반환됨
class URLDataset(Dataset):
    def __init__(self, csv_path, vocab=None, build_new_vocab=False):
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        url = self.df.iloc[idx]["url"]
        label = self.df.iloc[idx]["label"]

        x = encode_url(url, self.vocab)

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.float32)

        return x, y