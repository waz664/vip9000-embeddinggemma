#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np

from npu_embed import embed_text


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / np.sum(exp)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the tiny Radxa Cubie A7S vector index.")
    parser.add_argument("query", nargs="?", help="Search text. If omitted, prompt interactively.")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--verbose-npu", action="store_true")
    args = parser.parse_args()

    query = args.query or input("search> ").strip()
    here = Path(__file__).resolve().parent
    index_dir = here / args.index_dir
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    matrix = np.load(index_dir / "embeddings.npy")

    q = embed_text(query, index_dir / "work", "query", verbose=args.verbose_npu)
    scores = matrix @ q
    order = np.argsort(scores)[::-1][: args.top_k]
    probabilities = softmax(scores[order] * args.temperature)

    print(f"query={query!r}")
    for rank, (idx, prob) in enumerate(zip(order, probabilities), start=1):
        chunk = chunks[int(idx)]
        score = float(scores[int(idx)])
        preview = " ".join(chunk["text"].split())
        if len(preview) > 420:
            preview = preview[:417] + "..."
        print()
        print(f"{rank}. probability={float(prob):.3f} cosine={score:.4f}")
        print(f"   source={chunk['url']}")
        print(f"   text={preview}")

    if not math.isclose(float(np.linalg.norm(q)), 1.0, rel_tol=1e-4):
        print("warning: query embedding was not unit length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
