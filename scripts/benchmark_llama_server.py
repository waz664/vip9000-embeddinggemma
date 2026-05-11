#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request


def post_json(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible llama.cpp chat endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8081/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-0.6b-powervr")
    parser.add_argument("--prompt", default="What is the capital of France? Answer briefly.")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.2,
    }
    for i in range(args.runs):
        t0 = time.perf_counter()
        result = post_json(args.url, payload, args.timeout)
        wall = time.perf_counter() - t0
        timings = result.get("timings", {})
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        print(
            f"run={i + 1} wall={wall:.2f}s "
            f"prompt_ms={timings.get('prompt_ms')} predicted_ms={timings.get('predicted_ms')} "
            f"predicted_per_second={timings.get('predicted_per_second')}"
        )
        print(f"answer={content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
