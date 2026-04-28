# 모델 검사 테스트용.
# url 하나를 넣으면 score/label을 출력합니다.
# (이슈 #3) HybridCharCNN: URL-derived feature는 직접 계산.
import argparse
import torch

from config import DEVICE, MODEL_PATH, THRESHOLD, FEATURE_COLS
from dataset import load_vocab, encode_url
from model import HybridCharCNN
from features import load_norm, assemble_inference_vector
from explain import get_xai_log


def predict_url(url: str, use_xai: bool = False) -> dict:
    vocab = load_vocab()  # !!예측 시에 새 vocab 만들면 안 됨
    mean, std = load_norm()

    # 모델 구조 생성
    model = HybridCharCNN(
        vocab_size=len(vocab),
        num_features=len(FEATURE_COLS),
    ).to(DEVICE)

    # 학습된 모델 가중치 로드
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # URL 전처리
    encoded = encode_url(url, vocab)
    x = torch.tensor([encoded], dtype=torch.long).to(DEVICE)

    # tabular feature: URL은 직접 계산 / HTML은 train mean으로 채워 정규화
    feat_vec = assemble_inference_vector(url, mean, std)
    f = torch.from_numpy(feat_vec).unsqueeze(0).to(DEVICE)

    # 예측
    with torch.no_grad():
        logit = model(x, f)
        score = torch.sigmoid(logit).item()

    result = {
        "url": url,
        "score": round(score, 4),
        "label": "malicious" if score >= THRESHOLD else "normal",
    }

    if use_xai:
        # HybridCharCNN.forward(x_url, x_tab)이라 captum에 x_tab을 함께 넘겨야 함.
        # 의도적으로 IG는 char embedding에만 적용 (baseline=zeros_like(x_url)).
        # tabular feature는 attribution 대상이 아니라 고정값으로 통과시킨다 —
        # XAI 결과가 char-level에 흐릿하게 나오면 모델이 tabular 분기에 더 의존한다는 신호.
        result["xai"] = get_xai_log(
            model=model,
            x=x,
            url=url,
            additional_forward_args=(f,),
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--xai", action="store_true")
    args = parser.parse_args()

    result = predict_url(args.url, use_xai=args.xai)
    print(result)
