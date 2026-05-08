#!/usr/bin/env python3
import argparse
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import requests

from npu_embed import embed_text


DEFAULT_URLS = [
    "https://radxa.com/products/cubie/a7s/",
    "https://docs.radxa.com/en/cubie/a7s",
    "https://docs.radxa.com/en/cubie/a7s/getting-started/quickly-started",
    "https://docs.radxa.com/en/cubie/a7s/hardware-use/hardware-info",
    "https://docs.radxa.com/en/cubie/a7s/download",
]


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "svg", "nav", "footer", "header", "button"}:
            self.skip += 1
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


def html_to_text(raw_html: str) -> str:
    parser = TextExtractor()
    parser.feed(raw_html)
    text = html.unescape(" ".join(parser.parts))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"Skip to main content", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(Overview|Documentation|Downloads|FAQs|Support|Accessories|Buy)(\s+\1)+", r"\1", text)
    return text.strip()


def fetch_page(url: str) -> str:
    response = requests.get(url, timeout=30, headers={"User-Agent": "radxa-npu-rag-demo/1.0"})
    response.raise_for_status()
    return html_to_text(response.text)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a tiny directory-backed vector index with the Radxa NPU.")
    parser.add_argument("--index-dir", default="index")
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument("--chunk-words", type=int, default=90)
    parser.add_argument("--overlap-words", type=int, default=20)
    parser.add_argument("--verbose-npu", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    index_dir = here / args.index_dir
    work_dir = index_dir / "work"
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for url in DEFAULT_URLS:
        print(f"fetch {url}", flush=True)
        chunks.extend(chunk_text(fetch_page(url), url, args.chunk_words, args.overlap_words))

    chunks = chunks[: args.max_chunks]
    if not chunks:
        raise RuntimeError("no source chunks were produced")

    vectors = []
    for i, chunk in enumerate(chunks):
        print(f"embed {i + 1}/{len(chunks)}: {chunk['text'][:72]}...", flush=True)
        vectors.append(embed_text(chunk["text"], work_dir, f"doc_{i:03d}", verbose=args.verbose_npu))

    matrix = np.vstack(vectors).astype(np.float32)
    np.save(index_dir / "embeddings.npy", matrix)
    (index_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    (index_dir / "sources.json").write_text(json.dumps(DEFAULT_URLS, indent=2), encoding="utf-8")

    print(f"wrote {len(chunks)} chunks", flush=True)
    print(f"index_dir={index_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
