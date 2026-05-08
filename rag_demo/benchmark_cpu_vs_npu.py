#!/usr/bin/env python3
import argparse
import resource
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece as spm
import tensorflow as tf

from npu_embed import embed_text


TFLITE_MODEL = "/home/radxa/embeddinggemma/embeddinggemma-300M_seq1024_mixed-precision.tflite"
TOKENIZER = "/home/radxa/embeddinggemma/tokenizer.model"


def cpu_seconds(kind: int) -> float:
    usage = resource.getrusage(kind)
    return float(usage.ru_utime + usage.ru_stime)


def token_ids(text: str, seq_len: int) -> np.ndarray:
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
    ids = [2] + sp.encode(text, out_type=int)[: seq_len - 2] + [1]
    ids.extend([0] * (seq_len - len(ids)))
    return np.asarray(ids, dtype=np.int32).reshape(1, seq_len)


def run_cpu_tflite(text: str, threads: int) -> dict:
    ids = token_ids(text, 1024)

    t0 = time.perf_counter()
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL, num_threads=threads)
    create_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    interpreter.allocate_tensors()
    allocate_s = time.perf_counter() - t0

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, ids)

    cpu0 = cpu_seconds(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    interpreter.invoke()
    invoke_s = time.perf_counter() - t0
    cpu_s = cpu_seconds(resource.RUSAGE_SELF) - cpu0

    embedding = interpreter.get_tensor(output_index).reshape(-1).astype(np.float32)
    embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
    return {
        "name": f"cpu_tflite_threads_{threads}",
        "create_s": create_s,
        "allocate_s": allocate_s,
        "invoke_s": invoke_s,
        "wall_s": create_s + allocate_s + invoke_s,
        "cpu_s": cpu_s,
        "embedding": embedding,
    }


def run_npu(text: str, work_dir: Path, verbose: bool) -> dict:
    self0 = cpu_seconds(resource.RUSAGE_SELF)
    child0 = cpu_seconds(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    embedding = embed_text(text, work_dir, "bench_npu", verbose=verbose)
    wall_s = time.perf_counter() - t0
    cpu_s = (cpu_seconds(resource.RUSAGE_SELF) - self0) + (cpu_seconds(resource.RUSAGE_CHILDREN) - child0)
    return {
        "name": "npu_viplite_masked_bias_hidden_fp32_seq128",
        "wall_s": wall_s,
        "cpu_s": cpu_s,
        "embedding": embedding,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def print_result(result: dict, baseline: Optional[np.ndarray] = None) -> None:
    cpu_pct = 100.0 * result["cpu_s"] / max(result["wall_s"], 1e-12)
    print(f"{result['name']}:")
    if "create_s" in result:
        print(f"  create_s={result['create_s']:.3f} allocate_s={result['allocate_s']:.3f} invoke_s={result['invoke_s']:.3f}")
    print(f"  wall_s={result['wall_s']:.3f}")
    print(f"  cpu_s={result['cpu_s']:.3f}")
    print(f"  avg_cpu_percent={cpu_pct:.1f}")
    print(f"  norm={float(np.linalg.norm(result['embedding'])):.6f}")
    if baseline is not None:
        print(f"  cosine_vs_cpu_tflite_threads_1={cosine(result['embedding'], baseline):.6f}")
    print("  first8=" + " ".join(f"{v:.6f}" for v in result["embedding"][:8]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CPU-only TFLite vs VIP9000 NPU embedding latency/load.")
    parser.add_argument("text", nargs="?", default="does Cubie A7S support NVMe storage?")
    parser.add_argument("--threads", default="1,4,8", help="Comma-separated CPU TFLite thread counts.")
    parser.add_argument("--verbose-npu", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    work_dir = here / "bench_work"
    thread_counts = [int(x) for x in args.threads.split(",") if x.strip()]

    print(f"text={args.text!r}")
    print("note=CPU TFLite is the original fixed seq1024 model; NPU path is the masked/bias seq128 transformer NBG.")
    print()

    cpu_results = [run_cpu_tflite(args.text, threads) for threads in thread_counts]
    baseline = cpu_results[0]["embedding"]
    for result in cpu_results:
        print_result(result, baseline)
        print()

    npu_result = run_npu(args.text, work_dir, args.verbose_npu)
    print_result(npu_result, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
