#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request


def post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the local VIP9000 RAG WebUI endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8080/api/chat")
    parser.add_argument("--query", default="Does the Cubie A7S support NVMe?")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    for i in range(args.runs):
        t0 = time.perf_counter()
        result = post_json(args.url, {"query": args.query}, args.timeout)
        wall = time.perf_counter() - t0
        timing = result.get("timing", {})
        print(
            f"run={i + 1} "
            f"wall={wall:.2f}s "
            f"embedding={timing.get('embedding_s', 0):.4f}s "
            f"llm={timing.get('llm_s', 0):.2f}s "
            f"total={timing.get('total_s', 0):.2f}s "
            f"embedding_cache_hit={result.get('embedding_cache_hit')} "
            f"response_cache_hit={result.get('response_cache_hit')} "
            f"used_kb={result.get('used_kb')} "
            f"model={result.get('model')}"
        )
        print(f"answer={result.get('answer', '').strip()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
