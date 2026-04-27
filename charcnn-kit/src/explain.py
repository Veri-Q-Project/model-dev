# 로드된 model과 인코딩된 입력 x를 받아 로그 설명을 반환하는 XAI 기능
# predict.py에서만 호출, 설명 외 기능 없음.
# 정상적으로 작동한다면 특별히 수정할 필요가 없습니다!!
# 해당 파일에서 문제가 발생하면 삭제하셔도 무방합니다.
import torch
from captum.attr import LayerIntegratedGradients


def get_xai_log(model, x, url: str, top_k: int = 10) -> dict:

    lig = LayerIntegratedGradients(model, model.embedding)

    attributions = lig.attribute(
        inputs=x,
        baselines=torch.zeros_like(x)
    )

    # [1, MAX_LEN, EMBED_DIM] -> [MAX_LEN]
    char_scores = attributions.sum(dim=2).squeeze(0)

    abs_scores = char_scores.abs()
    top_indices = torch.topk(
        abs_scores,
        k=min(top_k, len(url))
    ).indices.tolist()

    highlights = []

    for idx in sorted(top_indices):
        if idx >= len(url):
            continue

        highlights.append({
            "position": idx,
            "char": url[idx],
            "score": round(char_scores[idx].item(), 6)
        })

    return {
        "method": "LayerIntegratedGradients",
        "top_chars": highlights
    }