# 모델 검사 테스트용.
# url 하나를 넣으면 score/label을 출력합니다.
import argparse
import torch

from config import DEVICE, MODEL_PATH, THRESHOLD
from dataset import load_vocab, encode_url
from model import CharCNN
from explain import get_xai_log



def predict_url(url: str, use_xai: bool = False) -> dict:
    vocab = load_vocab() # !!예측 시에 새 vocab 만들면 안 됨

    # 모델 구조 생성
    model = CharCNN(vocab_size=len(vocab)).to(DEVICE)
    
    # 학습된 모델 가중치 로드
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 예측 모드로 전환
    model.eval()

    # 분석할 url 전처리
    encoded = encode_url(url, vocab)
    x = torch.tensor([encoded], dtype=torch.long).to(DEVICE)

    # 예측 실행
    with torch.no_grad():
        logit = model(x)
        score = torch.sigmoid(logit).item()

    result = {
        "url": url,
        "score": round(score, 4),
        "label": "malicious" if score >= THRESHOLD else "normal"
    }

    if use_xai:
        result["xai"] = get_xai_log(
            model=model,
            x=x,
            url=url
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--xai", action="store_true")
    args = parser.parse_args()

    result = predict_url(args.url, use_xai=args.xai)
    print(result)