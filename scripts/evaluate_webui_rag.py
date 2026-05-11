#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


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


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def evaluate_case(case: dict, result: dict, wall_s: float) -> dict:
    hits = result.get("hits") or result.get("candidate_hits") or []
    top = hits[0] if hits else {}
    answer = str(result.get("answer", ""))
    haystack = " ".join(
        [
            answer,
            str(top.get("url", "")),
            str(top.get("text", "")),
        ]
    ).lower()
    expected_source = str(case.get("expected_source_contains", "")).lower()
    keywords = [str(k).lower() for k in case.get("expected_keywords", [])]
    normalized_source = expected_source.replace("-", "").replace("/", "")
    normalized_url = str(top.get("url", "")).lower().replace("-", "").replace("/", "")
    source_ok = (not expected_source) or normalized_source in normalized_url
    keyword_hits = [kw for kw in keywords if kw in haystack]
    keywords_ok = len(keyword_hits) == len(keywords)
    used_kb = bool(result.get("used_kb"))
    passed = bool(used_kb and source_ok and keywords_ok)
    timing = result.get("timing", {})
    return {
        "id": case.get("id"),
        "query": case.get("query"),
        "passed": passed,
        "used_kb": used_kb,
        "source_ok": source_ok,
        "keywords_ok": keywords_ok,
        "keyword_hits": keyword_hits,
        "expected_keywords": keywords,
        "top_url": top.get("url"),
        "top_cosine": top.get("cosine"),
        "wall_s": wall_s,
        "embedding_s": timing.get("embedding_s"),
        "llm_s": timing.get("llm_s"),
        "total_s": timing.get("total_s"),
        "embedding_cache_hit": result.get("embedding_cache_hit"),
        "response_cache_hit": result.get("response_cache_hit"),
        "answer": answer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local RAG WebUI against fixed QA checks.")
    parser.add_argument("--url", default="http://127.0.0.1:8080/api/chat")
    parser.add_argument("--cases", type=Path, default=Path("eval/rag_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/webui_rag_eval.json"))
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases.")
    parser.add_argument("--timeout", type=int, default=360)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    results = []
    for case in cases:
        t0 = time.perf_counter()
        raw = post_json(args.url, {"query": case["query"]}, args.timeout)
        wall_s = time.perf_counter() - t0
        item = evaluate_case(case, raw, wall_s)
        results.append(item)
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"{status} {item['id']} wall={wall_s:.2f}s "
            f"embed={item['embedding_s']:.4f}s llm={item['llm_s']:.2f}s "
            f"kb={item['used_kb']} cache={item['embedding_cache_hit']}/{item['response_cache_hit']} "
            f"url={item['top_url']}"
        )

    passed = sum(1 for item in results if item["passed"])
    totals = [float(item["total_s"]) for item in results if item["total_s"] is not None]
    summary = {
        "cases": len(results),
        "passed": passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "median_total_s": statistics.median(totals) if totals else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"summary cases={summary['cases']} passed={summary['passed']} "
        f"pass_rate={summary['pass_rate']:.2%} median_total_s={summary['median_total_s']}"
    )
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
