#!/usr/bin/env python3
import json
import hashlib
import cgi
import html
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

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
CONTEXT_CHARS = int(os.environ.get("VIP9000_RAG_CONTEXT_CHARS", "1000"))
KB_MIN_COSINE = float(os.environ.get("VIP9000_RAG_MIN_COSINE", "0.35"))
QUERY_CACHE = os.environ.get("VIP9000_RAG_QUERY_CACHE", "1") != "0"
RESPONSE_CACHE = os.environ.get("VIP9000_RAG_RESPONSE_CACHE", "1") != "0"
WEB_SEARCH = os.environ.get("VIP9000_RAG_WEB_SEARCH", "1") != "0"
INGEST_MAX_PAGES = int(os.environ.get("VIP9000_RAG_INGEST_MAX_PAGES", "8"))
INGEST_MAX_CHUNKS = int(os.environ.get("VIP9000_RAG_INGEST_MAX_CHUNKS", "24"))
PORT = int(os.environ.get("VIP9000_RAG_PORT", "8080"))
INDEX_LOCK = threading.Lock()
STATS_LOCK = threading.Lock()
STATS = {
    "requests": 0,
    "kb_used": 0,
    "web_searches": 0,
    "ingested_sources": 0,
    "ingested_chunks": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "embedding_s": 0.0,
    "llm_s": 0.0,
}

sys.path.insert(0, str(MODEL_DIR))
from embed_text_bias_hidden_npu import embed_text  # noqa: E402


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = dict(attrs)
        if tag in {"script", "style", "svg", "nav", "footer", "header", "button"}:
            self.skip += 1
        if tag == "a" and attrs_dict.get("href"):
            self.links.append(attrs_dict["href"])
        if tag in {"p", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "nav", "footer", "header", "button"} and self.skip:
            self.skip -= 1
        if tag in {"p", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


class SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._in_link = False
        self._href = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = dict(attrs)
        if tag == "a" and "result__a" in attrs_dict.get("class", ""):
            self._in_link = True
            self._href = attrs_dict.get("href", "")
            self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            title = clean_text(" ".join(self._text))
            url = normalize_duckduckgo_url(self._href)
            if title and url:
                self.results.append({"title": title, "url": url, "snippet": ""})
            self._in_link = False

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._text.append(data)


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text_and_links(raw_html: str) -> tuple[str, list[str]]:
    parser = TextExtractor()
    parser.feed(raw_html)
    text = clean_text(" ".join(parser.parts))
    text = re.sub(r"Skip to main content", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(Overview|Documentation|Downloads|FAQs|Support|Accessories|Buy)(\s+\1)+", r"\1", text)
    return text.strip(), parser.links


def fetch_url(url: str) -> tuple[str, list[str], str]:
    request = urllib.request.Request(url, headers={"User-Agent": "vip9000-rag-webui/0.2"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "")
        final_url = response.geturl()
        raw = response.read(2_000_000)
    text = raw.decode("utf-8", errors="replace")
    if "html" in content_type or "<html" in text[:512].lower():
        extracted, links = html_to_text_and_links(text)
        return extracted, links, final_url
    return clean_text(text), [], final_url


def chunk_text(source: str, url: str, max_words: int = 90, overlap_words: int = 20) -> list[dict]:
    words = source.split()
    chunks = []
    step = max(max_words - overlap_words, 1)
    for start in range(0, len(words), step):
        piece = " ".join(words[start : start + max_words]).strip()
        if len(piece) < 120:
            continue
        chunks.append({"url": url, "text": piece})
    return chunks


def load_index() -> tuple[list[dict], np.ndarray]:
    with INDEX_LOCK:
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


def response_cache_path(query: str, hits: list[dict], used_kb: bool, web_hits: Optional[list[dict]] = None) -> Path:
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
        "cache_version": 2,
        "index": index_fingerprint(),
        "provider": LLM_PROVIDER,
        "model": LLAMA_CPP_MODEL if LLM_PROVIDER == "llama_cpp" else OLLAMA_MODEL,
        "top_k": TOP_K,
        "context_chars": CONTEXT_CHARS,
        "min_cosine": KB_MIN_COSINE,
        "used_kb": used_kb,
        "sources": sources,
        "web_sources": [
            {
                "rank": hit.get("rank"),
                "url": hit.get("url"),
                "title": hit.get("title"),
            }
            for hit in (web_hits or [])
        ],
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return WORK_DIR / "response_cache" / f"{key}.json"


def clear_response_cache() -> None:
    cache_dir = WORK_DIR / "response_cache"
    if not cache_dir.exists():
        return
    for path in cache_dir.glob("*.json"):
        try:
            path.unlink()
        except OSError:
            pass


def normalize_duckduckgo_url(href: str) -> str:
    parsed = urlparse(href)
    if parsed.path == "/l/":
        qs = parse_qs(parsed.query)
        if qs.get("uddg"):
            return qs["uddg"][0]
    return href


def web_search(query: str, limit: int = 3) -> tuple[list[dict], float]:
    if not WEB_SEARCH:
        return [], 0.0
    t0 = time.perf_counter()
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": "vip9000-rag-webui/0.2"})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read(500_000).decode("utf-8", errors="replace")
    parser = SearchParser()
    parser.feed(raw)
    hits = []
    seen = set()
    for item in parser.results:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        hits.append({"rank": len(hits) + 1, **item})
        if len(hits) >= limit:
            break
    return hits, time.perf_counter() - t0


def should_web_search(query: str, used_kb: bool, allow_web: bool) -> bool:
    if not WEB_SEARCH or not allow_web:
        return False
    lowered = query.lower()
    triggers = ("web", "search", "latest", "current", "today", "news", "download", "release")
    return (not used_kb) or any(term in lowered for term in triggers)


def add_stats(**kwargs) -> None:
    with STATS_LOCK:
        for key, value in kwargs.items():
            if key in STATS:
                STATS[key] += value


def stats_snapshot() -> dict:
    with STATS_LOCK:
        return dict(STATS)


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


def build_messages(query: str, hits: list[dict], web_hits: Optional[list[dict]] = None) -> tuple[list[dict], bool]:
    use_kb = bool(hits and hits[0]["cosine"] >= KB_MIN_COSINE)
    web_hits = web_hits or []
    if use_kb:
        context_parts = [
            f"[{hit['rank']}] Source: {hit['url']}\n{hit['text'][:CONTEXT_CHARS]}" for hit in hits
        ]
        context_parts.extend(
            f"[W{hit['rank']}] Web Source: {hit['url']}\n{hit['title']}" for hit in web_hits
        )
        context = "\n\n".join(context_parts)
        system = (
            "Answer using only the retrieved context and web search results. "
            "Be concise. Cite KB sources with bracket numbers like [1] and web sources like [W1]."
        )
        prompt = f"Retrieved context:\n{context}\n\nQuestion: {query}"
    elif web_hits:
        context = "\n\n".join(
            f"[W{hit['rank']}] Source: {hit['url']}\n{hit['title']}" for hit in web_hits
        )
        system = (
            "Answer using the web search results when relevant. "
            "Be concise. Cite web sources with bracket numbers like [W1]."
        )
        prompt = f"Web search results:\n{context}\n\nQuestion: {query}"
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


def llm_answer(query: str, hits: list[dict], web_hits: Optional[list[dict]] = None) -> tuple[str, float, bool, str, dict]:
    messages, use_kb = build_messages(query, hits, web_hits)
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
        usage = result.get("usage", {}) if isinstance(result.get("usage", {}), dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        return content.strip(), elapsed, use_kb, LLAMA_CPP_MODEL, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0),
        }
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
    prompt_tokens = int(result.get("prompt_eval_count", 0) or 0)
    completion_tokens = int(result.get("eval_count", 0) or 0)
    return result.get("message", {}).get("content", "").strip(), elapsed, use_kb, OLLAMA_MODEL, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def cached_llm_answer(query: str, hits: list[dict], web_hits: Optional[list[dict]] = None) -> tuple[str, float, bool, str, bool, dict]:
    _, use_kb = build_messages(query, hits, web_hits)
    cache_path = response_cache_path(query, hits, use_kb, web_hits)
    if RESPONSE_CACHE and cache_path.exists():
        t0 = time.perf_counter()
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return (
            str(cached.get("answer", "")).strip(),
            time.perf_counter() - t0,
            bool(cached.get("used_kb", use_kb)),
            str(cached.get("model", LLAMA_CPP_MODEL if LLM_PROVIDER == "llama_cpp" else OLLAMA_MODEL)),
            True,
            cached.get("usage", {}) if isinstance(cached.get("usage", {}), dict) else {},
        )

    answer, elapsed, used_kb, model_name, usage = llm_answer(query, hits, web_hits)
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
                    "usage": usage,
                    "created_at": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(cache_path)
    return answer, elapsed, used_kb, model_name, False, usage


def ingest_chunks(new_chunks: list[dict]) -> tuple[int, float]:
    if not new_chunks:
        return 0, 0.0
    new_chunks = new_chunks[:INGEST_MAX_CHUNKS]
    vectors = []
    t0 = time.perf_counter()
    for i, chunk in enumerate(new_chunks):
        text = " ".join(chunk["text"].split())
        chunk["text"] = text
        vectors.append(embed_text(text, WORK_DIR / "ingest", f"doc_{int(time.time())}_{i:03d}", verbose=False))
    embed_s = time.perf_counter() - t0
    with INDEX_LOCK:
        chunks = json.loads((RAG_DIR / "chunks.json").read_text(encoding="utf-8"))
        matrix = np.load(RAG_DIR / "embeddings.npy")
        chunks.extend(new_chunks)
        matrix = np.vstack([matrix, np.vstack(vectors).astype(np.float32)])
        tmp_chunks = RAG_DIR / "chunks.tmp.json"
        tmp_matrix = RAG_DIR / "embeddings.tmp.npy"
        tmp_chunks.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        np.save(tmp_matrix, matrix)
        tmp_chunks.replace(RAG_DIR / "chunks.json")
        tmp_matrix.replace(RAG_DIR / "embeddings.npy")
    clear_response_cache()
    add_stats(ingested_sources=1, ingested_chunks=len(new_chunks), embedding_s=embed_s)
    return len(new_chunks), embed_s


def same_site_links(base_url: str, links: list[str]) -> list[str]:
    base = urlparse(base_url)
    out = []
    seen = {base_url}
    for href in links:
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != base.netloc:
            continue
        cleaned = parsed._replace(fragment="").geturl()
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= INGEST_MAX_PAGES - 1:
            break
    return out


def ingest_url(url: str) -> dict:
    text, links, final_url = fetch_url(url)
    pages = [(final_url, text)]
    for child_url in same_site_links(final_url, links):
        try:
            child_text, _, child_final = fetch_url(child_url)
            pages.append((child_final, child_text))
        except Exception as exc:
            print(f"ingest child skipped {child_url}: {exc}", flush=True)
        if len(pages) >= INGEST_MAX_PAGES:
            break
    chunks = []
    for page_url, page_text in pages:
        chunks.extend(chunk_text(page_text, page_url))
    count, embed_s = ingest_chunks(chunks)
    return {"pages": len(pages), "chunks": count, "embedding_s": embed_s}


def ingest_file(name: str, data: bytes) -> dict:
    text = data.decode("utf-8", errors="replace")
    if "<html" in text[:1024].lower():
        text, _ = html_to_text_and_links(text)
    chunks = chunk_text(clean_text(text), f"upload:{name}")
    count, embed_s = ingest_chunks(chunks)
    return {"pages": 1, "chunks": count, "embedding_s": embed_s}


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
        if path == "/api/status":
            chunks, _ = load_index()
            payload = {
                "stats": stats_snapshot(),
                "index": {
                    "chunks": len(chunks),
                    "fingerprint": index_fingerprint(),
                },
                "provider": LLM_PROVIDER,
                "model": LLAMA_CPP_MODEL if LLM_PROVIDER == "llama_cpp" else OLLAMA_MODEL,
                "web_search_enabled": WEB_SEARCH,
            }
            self.send_json(200, payload)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/ingest":
            self.handle_ingest()
            return
        if path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            query = str(payload.get("query", "")).strip()
            allow_web = bool(payload.get("allow_web", False))
            if not query:
                self.send_json(400, {"error": "query is required"})
                return
            t0 = time.perf_counter()
            hits, embed_s, embedding_cache_hit = retrieve(query)
            used_kb_before_search = bool(hits and hits[0]["cosine"] >= KB_MIN_COSINE)
            web_hits = []
            web_s = 0.0
            if should_web_search(query, used_kb_before_search, allow_web):
                try:
                    web_hits, web_s = web_search(query)
                except Exception as exc:
                    print(f"web search failed: {exc}", flush=True)
                    web_hits = []
                    web_s = 0.0
            answer, llm_s, used_kb, model_name, response_cache_hit, usage = cached_llm_answer(query, hits, web_hits)
            total_s = time.perf_counter() - t0
            add_stats(
                requests=1,
                kb_used=1 if used_kb else 0,
                web_searches=1 if web_hits else 0,
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0) if not response_cache_hit else 0,
                completion_tokens=int(usage.get("completion_tokens", 0) or 0) if not response_cache_hit else 0,
                total_tokens=int(usage.get("total_tokens", 0) or 0) if not response_cache_hit else 0,
                embedding_s=embed_s,
                llm_s=llm_s,
            )
            self.send_json(
                200,
                {
                    "answer": answer,
                    "hits": hits if used_kb else [],
                    "candidate_hits": hits,
                    "web_hits": web_hits,
                    "used_kb": used_kb,
                    "min_cosine": KB_MIN_COSINE,
                    "timing": {
                        "embedding_s": embed_s,
                        "web_s": web_s,
                        "llm_s": llm_s,
                        "total_s": total_s,
                    },
                    "embedding_cache_hit": embedding_cache_hit,
                    "response_cache_hit": response_cache_hit,
                    "usage": usage,
                    "stats": stats_snapshot(),
                    "model": model_name,
                    "provider": LLM_PROVIDER,
                },
            )
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def handle_ingest(self) -> None:
        try:
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", "0"))
            if content_type.startswith("multipart/form-data"):
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                        "CONTENT_LENGTH": str(length),
                    },
                )
                if "file" not in form:
                    self.send_json(400, {"error": "file field is required"})
                    return
                item = form["file"]
                name = Path(getattr(item, "filename", "") or "upload.txt").name
                result = ingest_file(name, item.file.read())
            else:
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                url = str(payload.get("url", "")).strip()
                if not url:
                    self.send_json(400, {"error": "url is required"})
                    return
                result = ingest_url(url)
            self.send_json(200, {"ok": True, **result, "stats": stats_snapshot()})
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
