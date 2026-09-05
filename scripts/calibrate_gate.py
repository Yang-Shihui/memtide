"""Calibrate the predictive-coding gate thresholds for the current embedder.

The gate routes on surprise S = -log2(max_sim^2); the 0.5/2.5 bit defaults
were calibrated against DashScope text-embedding measurements. Different
embedding models have different similarity distributions, so after switching
embedders run this script: it embeds a built-in set of paraphrase pairs
(same fact, reworded, zh+en), cross-language equivalents and unrelated pairs,
prints the similarity distributions and SUGGESTS thresholds.

Usage (with the same env/config the deployment uses):
    python scripts/calibrate_gate.py            # reads EMBEDDING_BACKEND / LLM_API_KEY / DASHSCOPE_API_KEY directly from env
    source .env && python scripts/calibrate_gate.py  # current production embedder
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memtide.embeddings import cosine, make_embedder  # noqa: E402
from memtide.config import MemoryConfig, config_from_env  # noqa: E402

# (a, b, kind): kind = same (paraphrase of one fact) | cross (unrelated facts)
PAIRS = [
    # zh paraphrases
    ("我叫李雷，住在杭州", "我的名字叫李雷，家在杭州", "same"),
    ("我喜欢喝美式咖啡", "美式咖啡是我的最爱", "same"),
    ("我是一名后端工程师", "我的职业是后端开发工程师", "same"),
    ("我搬到上海了", "我把家搬到了上海", "same"),
    ("我最近在学 Rust", "我正在学习 Rust 这门语言", "same"),
    # en paraphrases
    ("My name is Alice and I love hiking", "Alice here — hiking is my favorite hobby", "same"),
    ("I work at Acme Corp as an engineer", "I'm an engineer employed by Acme Corp", "same"),
    ("The user lives in Hangzhou", "The user's home city is Hangzhou", "same"),
    # cross-language equivalents
    ("用户住在上海", "The user lives in Shanghai", "same"),
    ("我喜欢喝咖啡", "I love drinking coffee", "same"),
    ("我是一名设计师", "I work as a designer", "same"),
    # unrelated
    ("我叫李雷，住在杭州", "今天的会议改到下午三点", "cross"),
    ("我喜欢喝美式咖啡", "Rust 的所有权模型很难学", "cross"),
    ("我是一名后端工程师", "明天的天气据说有雨", "cross"),
    ("The user loves hiking", "Please send the invoice by Friday", "cross"),
    ("用户住在上海", "量子计算机使用量子比特运算", "cross"),
]


def _pctl(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    cfg = MemoryConfig(
        embedding_backend=os.environ.get("EMBEDDING_BACKEND", "auto"),
        embedding_base_url=os.environ.get("EMBEDDING_BASE_URL", ""),
        llm_base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        llm_api_key=os.environ.get("LLM_API_KEY"),
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
    )
    emb = make_embedder(cfg)
    print(f"embedder: {type(emb).__name__}  dim={getattr(emb, 'dim', '?')}\n")

    texts = [t for pair in PAIRS for t in pair[:2]]
    vecs = emb.embed_batch(texts)
    sims = {"same": [], "cross": []}
    for (a, b, kind), (ia, ib) in zip(PAIRS, [(i * 2, i * 2 + 1) for i in range(len(PAIRS))]):
        sim = cosine(vecs[ia], vecs[ib])
        sims[kind].append(sim)
        flag = ""
        # flag pairs that land on the wrong side of the current defaults
        cur_redundant = 0.5
        if kind == "same" and sim < 0.841:  # cos for S=0.5 bits
            flag = "  <- below current REJECT line"
        if kind == "cross" and sim > 0.177:  # cos for S=2.5 bits
            flag = "  <- above current NOVEL line"
        print(f"  [{kind:5}] cos={sim:6.3f}  {a[:28]!r} vs {b[:28]!r}{flag}")

    same_sorted = sorted(sims["same"])
    cross_sorted = sorted(sims["cross"])
    print(f"\nsame  n={len(same_sorted)}  p5={_pctl(same_sorted, .05):.3f} "
          f"p25={_pctl(same_sorted, .25):.3f} p50={_pctl(same_sorted, .5):.3f} "
          f"p75={_pctl(same_sorted, .75):.3f}")
    print(f"cross n={len(cross_sorted)}  p5={_pctl(cross_sorted, .05):.3f} "
          f"p50={_pctl(cross_sorted, .5):.3f} p75={_pctl(cross_sorted, .75):.3f} "
          f"p95={_pctl(cross_sorted, .95):.3f}")

    import math

    # suggested lines: keep 95% of paraphrases out of REJECT, let 95% of
    # unrelated facts stay above NOVEL
    s_redundant = -math.log2(max(_pctl(same_sorted, .05), 0.02) ** 2)
    s_novel = -math.log2(max(_pctl(cross_sorted, .95), 0.02) ** 2)
    print("\nsuggested MemoryConfig for THIS embedder:")
    print(json.dumps({"gate_redundant_bits": round(s_redundant, 2),
                      "gate_novel_bits": round(s_novel, 2)}, indent=2))


if __name__ == "__main__":
    main()
