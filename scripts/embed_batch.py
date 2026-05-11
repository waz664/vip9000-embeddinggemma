#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed one text per line with the VIP9000 EmbeddingGemma runtime.")
    parser.add_argument("--model-dir", default=str(Path.home() / "embeddinggemma_npu_seq128_bias_hidden_fp32"))
    parser.add_argument("--input", default="-", help="Input JSONL/text file, or '-' for stdin")
    parser.add_argument("--output", required=True, help="Output .npy matrix")
    parser.add_argument("--texts-json", help="Optional sidecar JSON path for input texts")
    parser.add_argument("--cache-dir", help="Persistent embedding cache directory")
    parser.add_argument("--work-dir", default="/tmp/vip9000-embed-batch")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    sys.path.insert(0, str(model_dir))
    if args.cache_dir:
        os.environ["EMBEDDINGGEMMA_EMBED_CACHE_DIR"] = args.cache_dir

    from embed_text_bias_hidden_npu import embed_text  # noqa: E402

    if args.input == "-":
        lines = sys.stdin.read().splitlines()
    else:
        lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    texts = [line.strip() for line in lines if line.strip()]
    if not texts:
        raise RuntimeError("no input texts")

    vectors = []
    work_dir = Path(args.work_dir)
    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        vectors.append(embed_text(text, work_dir, f"batch_{i:05d}", verbose=False))
        print(f"embedded {i + 1}/{len(texts)} in {time.perf_counter() - t0:.3f}s", file=sys.stderr, flush=True)

    matrix = np.vstack(vectors).astype(np.float32)
    np.save(args.output, matrix)
    if args.texts_json:
        Path(args.texts_json).write_text(json.dumps(texts, indent=2), encoding="utf-8")
    print(f"wrote {matrix.shape[0]} embeddings to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
