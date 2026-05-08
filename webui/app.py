#!/usr/bin/env python3
import json
import math
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


import os

MODEL_DIR = Path(os.environ.get("VIP9000_RAG_MODEL_DIR", "/home/radxa/embeddinggemma_npu_seq128_bias_hidden_fp32"))
RAG_DIR = MODEL_DIR / "rag_demo" / "index"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3:0.6b"
TOP_K = 2
PORT = int(os.environ.get("VIP9000_RAG_PORT", "8080"))

import sys

sys.path.insert(0, str(MODEL_DIR))
from embed_text_bias_hidden_npu import embed_text  # noqa: E402


def load_index() -> tuple[list[dict], np.ndarray]:
    chunks = json.loads((RAG_DIR / "chunks.json").read_text(encoding="utf-8"))
    matrix = np.load(RAG_DIR / "embeddings.npy")
    return chunks, matrix


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / np.sum(exp)


def retrieve(query: str) -> tuple[list[dict], float]:
    chunks, matrix = load_index()
    t0 = time.perf_counter()
    q = embed_text(query, Path("/home/radxa/rag_webui/work"), "query", verbose=False)
    embed_s = time.perf_counter() - t0
    scores = matrix @ q
    order = np.argsort(scores)[::-1][:TOP_K]
    probs = softmax(scores[order] * 20.0)
    hits = []
    for rank, (idx, prob) in enumerate(zip(order, probs), start=1):
        chunk = chunks[int(idx)]
        hits.append(
            {
                "rank": rank,
                "probability": float(prob),
                "cosine": float(scores[int(idx)]),
                "url": chunk["url"],
                "text": " ".join(chunk["text"].split()),
            }
        )
    return hits, embed_s


def ollama_answer(query: str, hits: list[dict]) -> tuple[str, float]:
    context = "\n\n".join(
        f"[{hit['rank']}] Source: {hit['url']}\n{hit['text'][:700]}" for hit in hits
    )
    system = (
        "Answer using only the retrieved context. "
        "Be concise. If the context is insufficient, say what is missing. "
        "Cite sources with bracket numbers like [1]."
    )
    prompt = (
        f"Retrieved context:\n{context}\n\n"
        f"Question: {query}"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 1024,
            "num_predict": 60,
            "num_thread": 4,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    return result.get("message", {}).get("content", "").strip(), time.perf_counter() - t0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = (Path(__file__).resolve().parent / "static" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            query = str(payload.get("query", "")).strip()
            if not query:
                self.send_json(400, {"error": "query is required"})
                return
            t0 = time.perf_counter()
            hits, embed_s = retrieve(query)
            answer, llm_s = ollama_answer(query, hits)
            self.send_json(
                200,
                {
                    "answer": answer,
                    "hits": hits,
                    "timing": {
                        "embedding_s": embed_s,
                        "llm_s": llm_s,
                        "total_s": time.perf_counter() - t0,
                    },
                    "model": OLLAMA_MODEL,
                },
            )
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})


def main() -> int:
    if not (RAG_DIR / "chunks.json").exists() or not (RAG_DIR / "embeddings.npy").exists():
        raise RuntimeError(f"missing index under {RAG_DIR}")
    if not math.isfinite(float(np.load(RAG_DIR / "embeddings.npy")[0, 0])):
        raise RuntimeError("index embeddings are invalid")
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"RAG WebUI listening on http://0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
