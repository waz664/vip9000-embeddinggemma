#!/usr/bin/env python3
import json
import hashlib
import math
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


MODEL_DIR = Path(os.environ.get("VIP9000_RAG_MODEL_DIR", str(Path.home() / "embeddinggemma_npu_seq128_bias_hidden_fp32")))
RAG_DIR = MODEL_DIR / "rag_demo" / "index"
WORK_DIR = Path(os.environ.get("VIP9000_RAG_WORK_DIR", str(MODEL_DIR / "webui_work")))
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "qwen3:0.6b"
LLM_PROVIDER = os.environ.get("VIP9000_RAG_LLM_PROVIDER", "ollama").strip().lower()
LLAMA_CPP_URL = os.environ.get("VIP9000_RAG_LLAMA_CPP_URL", "http://127.0.0.1:8081/v1/chat/completions")
LLAMA_CPP_MODEL = os.environ.get("VIP9000_RAG_LLAMA_CPP_MODEL", "qwen3-0.6b-powervr")
TOP_K = int(os.environ.get("VIP9000_RAG_TOP_K", "1"))
CONTEXT_CHARS = int(os.environ.get("VIP9000_RAG_CONTEXT_CHARS", "450"))
KB_MIN_COSINE = float(os.environ.get("VIP9000_RAG_MIN_COSINE", "0.35"))
QUERY_CACHE = os.environ.get("VIP9000_RAG_QUERY_CACHE", "1") != "0"
RESPONSE_CACHE = os.environ.get("VIP9000_RAG_RESPONSE_CACHE", "1") != "0"
PORT = int(os.environ.get("VIP9000_RAG_PORT", "8080"))

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


def query_cache_path(query: str) -> Path:
    key = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
    return WORK_DIR / "query_cache" / f"{key}.npy"


def index_fingerprint() -> str:
    h = hashlib.sha256()
    for name in ("chunks.json", "embeddings.npy"):
        path = RAG_DIR / name
        stat = path.stat()
        h.update(name.encode("utf-8"))
        h.update(str(stat.st_mtime_ns).encode("ascii"))
        h.update(str(stat.st_size).encode("ascii"))
    return h.hexdigest()


def response_cache_path(query: str, hits: list[dict], used_kb: bool) -> Path:
    sources = [
        {
            "rank": hit.get("rank"),
            "url": hit.get("url"),
            "cosine": round(float(hit.get("cosine", 0.0)), 6),
            "text_sha256": hashlib.sha256(str(hit.get("text", "")).encode("utf-8")).hexdigest(),
        }
        for hit in hits
    ] if used_kb else []
    payload = {
        "query": query.strip(),
        "index": index_fingerprint(),
        "provider": LLM_PROVIDER,
        "model": LLAMA_CPP_MODEL if LLM_PROVIDER == "llama_cpp" else OLLAMA_MODEL,
        "top_k": TOP_K,
        "context_chars": CONTEXT_CHARS,
        "min_cosine": KB_MIN_COSINE,
        "used_kb": used_kb,
        "sources": sources,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return WORK_DIR / "response_cache" / f"{key}.json"


def embed_query(query: str) -> tuple[np.ndarray, float, bool]:
    cache_path = query_cache_path(query)
    if QUERY_CACHE and cache_path.exists():
        t0 = time.perf_counter()
        return np.load(cache_path), time.perf_counter() - t0, True

    t0 = time.perf_counter()
    q = embed_text(query, WORK_DIR, "query", verbose=False)
    elapsed = time.perf_counter() - t0

    if QUERY_CACHE:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp.npy")
        np.save(tmp_path, q)
        tmp_path.replace(cache_path)

    return q, elapsed, False


def retrieval_query(query: str) -> str:
    expanded = query
    lowered = query.lower()
    additions = []
    if "processor" in lowered and "soc" not in lowered:
        additions.extend(["SoC", "CPU"])
    if "storage" in lowered and "nvme" not in lowered:
        additions.extend(["NVMe", "PCIe"])
    if "memory" in lowered and "lpddr" not in lowered:
        additions.extend(["RAM", "LPDDR5"])
    if "display" in lowered and "usb-c" not in lowered:
        additions.extend(["USB-C", "DisplayPort"])
    if additions:
        expanded = f"{query} {' '.join(additions)}"
    return expanded


def lexical_boosts(query: str, chunks: list[dict]) -> np.ndarray:
    lowered = query.lower()
    terms = []
    if "memory" in lowered or "ram" in lowered or "lpddr" in lowered:
        terms.extend(["lpddr", "memory"])
    if "processor" in lowered or "soc" in lowered:
        terms.extend(["allwinner", "a733", "soc"])
    if "storage" in lowered or "nvme" in lowered:
        terms.extend(["nvme", "pcie"])
    if "display" in lowered or "usb-c" in lowered:
        terms.extend(["displayport", "usb-c"])

    boosts = np.zeros(len(chunks), dtype=np.float32)
    if not terms:
        return boosts
    for i, chunk in enumerate(chunks):
        text = f"{chunk.get('url', '')} {chunk.get('text', '')}".lower()
        hits = sum(1 for term in terms if term in text)
        if hits:
            boosts[i] = min(0.03 * hits, 0.09)
    return boosts


def retrieve(query: str) -> tuple[list[dict], float, bool]:
    chunks, matrix = load_index()
    expanded_query = retrieval_query(query)
    q, embed_s, cache_hit = embed_query(expanded_query)
    scores = (matrix @ q) + lexical_boosts(expanded_query, chunks)
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
    return hits, embed_s, cache_hit


def build_messages(query: str, hits: list[dict]) -> tuple[list[dict], bool]:
    use_kb = bool(hits and hits[0]["cosine"] >= KB_MIN_COSINE)
    if use_kb:
        context = "\n\n".join(
            f"[{hit['rank']}] Source: {hit['url']}\n{hit['text'][:CONTEXT_CHARS]}" for hit in hits
        )
        system = (
            "Answer using only the retrieved context. "
            "Be concise. Cite sources with bracket numbers like [1]."
        )
        prompt = f"Retrieved context:\n{context}\n\nQuestion: {query}"
    else:
        system = (
            "Answer normally and concisely. The local knowledgebase did not contain "
            "a relevant source, so do not cite KB sources."
        )
        prompt = query
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ], use_kb


def post_json(url: str, payload: dict, timeout_s: int = 240) -> tuple[dict, float]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    return result, time.perf_counter() - t0


def llm_answer(query: str, hits: list[dict]) -> tuple[str, float, bool, str]:
    messages, use_kb = build_messages(query, hits)
    if LLM_PROVIDER == "llama_cpp":
        payload = {
            "model": LLAMA_CPP_MODEL,
            "messages": messages,
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 80,
        }
        result, elapsed = post_json(LLAMA_CPP_URL, payload)
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip(), elapsed, use_kb, LLAMA_CPP_MODEL
    if LLM_PROVIDER != "ollama":
        raise RuntimeError(f"unsupported VIP9000_RAG_LLM_PROVIDER={LLM_PROVIDER!r}")
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 1024,
            "num_predict": 60,
            "num_thread": 4,
        },
    }
    result, elapsed = post_json(OLLAMA_URL, payload)
    return result.get("message", {}).get("content", "").strip(), elapsed, use_kb, OLLAMA_MODEL


def cached_llm_answer(query: str, hits: list[dict]) -> tuple[str, float, bool, str, bool]:
    _, use_kb = build_messages(query, hits)
    cache_path = response_cache_path(query, hits, use_kb)
    if RESPONSE_CACHE and cache_path.exists():
        t0 = time.perf_counter()
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return (
            str(cached.get("answer", "")).strip(),
            time.perf_counter() - t0,
            bool(cached.get("used_kb", use_kb)),
            str(cached.get("model", LLAMA_CPP_MODEL if LLM_PROVIDER == "llama_cpp" else OLLAMA_MODEL)),
            True,
        )

    answer, elapsed, used_kb, model_name = llm_answer(query, hits)
    if RESPONSE_CACHE:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "answer": answer,
                    "used_kb": used_kb,
                    "model": model_name,
                    "provider": LLM_PROVIDER,
                    "created_at": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(cache_path)
    return answer, elapsed, used_kb, model_name, False


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
            hits, embed_s, embedding_cache_hit = retrieve(query)
            answer, llm_s, used_kb, model_name, response_cache_hit = cached_llm_answer(query, hits)
            self.send_json(
                200,
                {
                    "answer": answer,
                    "hits": hits if used_kb else [],
                    "candidate_hits": hits,
                    "used_kb": used_kb,
                    "min_cosine": KB_MIN_COSINE,
                    "timing": {
                        "embedding_s": embed_s,
                        "llm_s": llm_s,
                        "total_s": time.perf_counter() - t0,
                    },
                    "embedding_cache_hit": embedding_cache_hit,
                    "response_cache_hit": response_cache_hit,
                    "model": model_name,
                    "provider": LLM_PROVIDER,
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
    print(f"RAG WebUI listening on http://0.0.0.0:{PORT} using {LLM_PROVIDER}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
