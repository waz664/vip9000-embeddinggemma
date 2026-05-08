#!/usr/bin/env python3
import argparse
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import tensorflow as tf

from build_index import DEFAULT_URLS, chunk_text, fetch_page


MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from embed_text_bias_hidden_npu import embed_text as embed_text_npu  # noqa: E402


TFLITE_MODEL = os.environ.get(
    "EMBEDDINGGEMMA_TFLITE",
    str(Path.home() / "embeddinggemma/embeddinggemma-300M_seq1024_mixed-precision.tflite"),
)
TOKENIZER = os.environ.get("EMBEDDINGGEMMA_TOKENIZER", str(MODEL_DIR / "tokenizer.model"))

DEFAULT_QUERIES = [
    "does Cubie A7S support NVMe storage?",
    "what is the NPU performance of the Cubie A7S?",
    "what CPU does the Radxa Cubie A7S use?",
    "does Cubie A7S support WiFi 6 and Bluetooth?",
    "what display output does Cubie A7S have?",
    "can Cubie A7S encode or decode 4K video?",
    "what camera interface is available on Cubie A7S?",
    "where can I download Cubie A7S images?",
    "what are the dimensions of the Cubie A7S board?",
    "what memory type does Cubie A7S use?",
    "does Cubie A7S have PCIe 3.0?",
    "what GPU is in the Allwinner A733?",
    "what CPU cores are in the Allwinner A733?",
    "what USB-C display capability does Cubie A7S provide?",
    "what operating system images are available for Cubie A7S?",
    "how do I get started with the Cubie A7S?",
    "what power connector does Cubie A7S use?",
    "does the Cubie A7S support MIPI CSI cameras?",
    "what is the RISC-V controller used for?",
    "what is the target use case for Cubie A7S?",
]

TARGETS = {
    "query_cos_mean": 0.93,
    "query_cos_min": 0.90,
    "top_k_overlap_mean": 0.85,
    "reference_top_mrr_in_npu": 0.70,
}


def cpu_seconds(kind: int) -> float:
    usage = resource.getrusage(kind)
    return float(usage.ru_utime + usage.ru_stime)


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1).astype(np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


class CpuTfliteEmbedder:
    def __init__(self, threads: int) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
        self.interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL, num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def token_ids(self, text: str) -> np.ndarray:
        ids = [2] + self.sp.encode(text, out_type=int)[:1022] + [1]
        ids.extend([0] * (1024 - len(ids)))
        return np.asarray(ids, dtype=np.int32).reshape(1, 1024)

    def embed(self, text: str) -> np.ndarray:
        self.interpreter.set_tensor(self.input_index, self.token_ids(text))
        self.interpreter.invoke()
        return normalize(self.interpreter.get_tensor(self.output_index))


def rankings(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    scores = matrix @ query
    return np.argsort(scores)[::-1]


def reciprocal_rank(reference_top: int, candidate_order: np.ndarray) -> float:
    hits = np.where(candidate_order == reference_top)[0]
    if hits.size == 0:
        return 0.0
    return 1.0 / float(hits[0] + 1)


def print_hit(chunks: list[dict], idx: int, prefix: str) -> None:
    text = " ".join(chunks[idx]["text"].split())
    if len(text) > 180:
        text = text[:177] + "..."
    print(f"  {prefix} #{idx:02d} {chunks[idx]['url']} :: {text}")


def main() -> int:
    parser = argparse.ArgumentParser(description="A/B retrieval quality eval: official CPU TFLite vs corrected NPU EmbeddingGemma.")
    parser.add_argument("--max-chunks", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", default="quality_eval")
    parser.add_argument("--verbose-npu", action="store_true")
    parser.add_argument("--target-report", action="store_true", help="Print pass/fail against built-in quality targets.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    out_dir = here / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "npu_work"

    chunks = []
    for url in DEFAULT_URLS:
        print(f"fetch {url}", flush=True)
        chunks.extend(chunk_text(fetch_page(url), url))
    chunks = chunks[: args.max_chunks]
    if not chunks:
        raise RuntimeError("no source chunks were produced")
    (out_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    print(f"chunks={len(chunks)} queries={len(DEFAULT_QUERIES)} top_k={args.top_k}", flush=True)
    cpu = CpuTfliteEmbedder(args.threads)

    cpu_wall0 = time.perf_counter()
    cpu_proc0 = cpu_seconds(resource.RUSAGE_SELF)
    cpu_doc = np.vstack([cpu.embed(chunk["text"]) for chunk in chunks]).astype(np.float32)
    cpu_query = np.vstack([cpu.embed(query) for query in DEFAULT_QUERIES]).astype(np.float32)
    cpu_wall = time.perf_counter() - cpu_wall0
    cpu_proc = cpu_seconds(resource.RUSAGE_SELF) - cpu_proc0

    npu_wall0 = time.perf_counter()
    npu_self0 = cpu_seconds(resource.RUSAGE_SELF)
    npu_child0 = cpu_seconds(resource.RUSAGE_CHILDREN)
    npu_doc = []
    for i, chunk in enumerate(chunks):
        print(f"npu doc {i + 1}/{len(chunks)}", flush=True)
        npu_doc.append(embed_text_npu(chunk["text"], work_dir, f"doc_{i:03d}", verbose=args.verbose_npu))
    npu_query = []
    for i, query in enumerate(DEFAULT_QUERIES):
        print(f"npu query {i + 1}/{len(DEFAULT_QUERIES)}: {query}", flush=True)
        npu_query.append(embed_text_npu(query, work_dir, f"query_{i:03d}", verbose=args.verbose_npu))
    npu_doc = np.vstack(npu_doc).astype(np.float32)
    npu_query = np.vstack(npu_query).astype(np.float32)
    npu_wall = time.perf_counter() - npu_wall0
    npu_cpu = (cpu_seconds(resource.RUSAGE_SELF) - npu_self0) + (cpu_seconds(resource.RUSAGE_CHILDREN) - npu_child0)

    np.save(out_dir / "cpu_doc_embeddings.npy", cpu_doc)
    np.save(out_dir / "cpu_query_embeddings.npy", cpu_query)
    np.save(out_dir / "npu_doc_embeddings.npy", npu_doc)
    np.save(out_dir / "npu_query_embeddings.npy", npu_query)

    doc_cos = [cosine(cpu_doc[i], npu_doc[i]) for i in range(len(chunks))]
    query_cos = [cosine(cpu_query[i], npu_query[i]) for i in range(len(DEFAULT_QUERIES))]

    overlaps = []
    top1_matches = 0
    rr_scores = []
    details = []
    for i, query in enumerate(DEFAULT_QUERIES):
        cpu_order = rankings(cpu_doc, cpu_query[i])
        npu_order = rankings(npu_doc, npu_query[i])
        cpu_top = set(int(x) for x in cpu_order[: args.top_k])
        npu_top = set(int(x) for x in npu_order[: args.top_k])
        overlap = len(cpu_top & npu_top) / float(args.top_k)
        rr = reciprocal_rank(int(cpu_order[0]), npu_order)
        overlaps.append(overlap)
        rr_scores.append(rr)
        top1_matches += int(cpu_order[0] == npu_order[0])
        details.append(
            {
                "query": query,
                "cpu_top": [int(x) for x in cpu_order[: args.top_k]],
                "npu_top": [int(x) for x in npu_order[: args.top_k]],
                "top_k_overlap": overlap,
                "reference_top_reciprocal_rank_in_npu": rr,
                "query_embedding_cosine": query_cos[i],
            }
        )

    summary = {
        "chunks": len(chunks),
        "queries": len(DEFAULT_QUERIES),
        "top_k": args.top_k,
        "cpu_wall_s": cpu_wall,
        "cpu_process_cpu_s": cpu_proc,
        "npu_wall_s": npu_wall,
        "npu_process_plus_child_cpu_s": npu_cpu,
        "doc_cos_mean": float(np.mean(doc_cos)),
        "doc_cos_min": float(np.min(doc_cos)),
        "query_cos_mean": float(np.mean(query_cos)),
        "query_cos_min": float(np.min(query_cos)),
        "top1_match_rate": top1_matches / float(len(DEFAULT_QUERIES)),
        "top_k_overlap_mean": float(np.mean(overlaps)),
        "reference_top_mrr_in_npu": float(np.mean(rr_scores)),
        "details": details,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("SUMMARY")
    for key in [
        "doc_cos_mean",
        "doc_cos_min",
        "query_cos_mean",
        "query_cos_min",
        "top1_match_rate",
        "top_k_overlap_mean",
        "reference_top_mrr_in_npu",
        "cpu_wall_s",
        "npu_wall_s",
        "npu_process_plus_child_cpu_s",
    ]:
        print(f"{key}={summary[key]:.4f}")
    print()
    print("TARGETS")
    for key, target in TARGETS.items():
        value = summary[key]
        status = "PASS" if value >= target else "FAIL"
        print(f"{key}>={target:.4f}: {value:.4f} {status}")
    print(f"summary_file={out_dir / 'summary.json'}")

    print()
    print("PER QUERY")
    for detail in details:
        print(f"query={detail['query']!r}")
        print(f"  query_cos={detail['query_embedding_cosine']:.4f} overlap@{args.top_k}={detail['top_k_overlap']:.2f} rr={detail['reference_top_reciprocal_rank_in_npu']:.2f}")
        print_hit(chunks, detail["cpu_top"][0], "cpu top1")
        print_hit(chunks, detail["npu_top"][0], "npu top1")
    if not math.isclose(float(np.linalg.norm(npu_query[0])), 1.0, rel_tol=1e-4):
        print("warning: NPU query embedding was not unit length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
