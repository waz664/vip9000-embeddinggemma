#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece as spm


ROOT = Path(__file__).resolve().parent
TOKENIZER = Path(os.environ.get("EMBEDDINGGEMMA_TOKENIZER", ROOT / "tokenizer.model"))
VPM_RUN = os.environ.get("VIP9000_VPM_RUN", str(Path.home() / "ai-sdk/examples/vpm_run/vpm_run"))
VIPLITE_LIB = os.environ.get(
    "VIP9000_VIPLIB",
    str(Path.home() / "ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0"),
)
SEQ_LEN = 128
HIDDEN = 768
VOCAB = 262144
BOS_ID = 2
EOS_ID = 1
PAD_ID = 0
MASK_BIAS = -10000.0
_SP = None
_TOKEN_TABLE = None
_DENSE_2 = None
_DENSE_3 = None


def token_ids(text: str) -> list[int]:
    global _SP
    if _SP is None:
        _SP = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    sp = _SP
    pieces = sp.encode(text, out_type=int)
    ids = [BOS_ID] + pieces[: SEQ_LEN - 2] + [EOS_ID]
    ids.extend([PAD_ID] * (SEQ_LEN - len(ids)))
    return ids


def token_table() -> np.memmap:
    global _TOKEN_TABLE
    if _TOKEN_TABLE is None:
        _TOKEN_TABLE = np.memmap(ROOT / "token_embedding_fp16.dat", dtype=np.float16, mode="r", shape=(VOCAB, HIDDEN))
    return _TOKEN_TABLE


def dense_tail_weights() -> tuple[np.ndarray, np.ndarray]:
    global _DENSE_2, _DENSE_3
    if _DENSE_2 is None:
        _DENSE_2 = np.load(ROOT / "dense_2_weight_f32.npy")
    if _DENSE_3 is None:
        _DENSE_3 = np.load(ROOT / "dense_3_weight_f32.npy")
    return _DENSE_2, _DENSE_3


def file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_key(text: str) -> str:
    payload = {
        "text": text,
        "seq_len": SEQ_LEN,
        "hidden": HIDDEN,
        "tokenizer": file_fingerprint(TOKENIZER),
        "network": file_fingerprint(ROOT / "network_binary.nb"),
        "token_embedding": file_fingerprint(ROOT / "token_embedding_fp16.dat"),
        "dense_2": file_fingerprint(ROOT / "dense_2_weight_f32.npy"),
        "dense_3": file_fingerprint(ROOT / "dense_3_weight_f32.npy"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def embedding_cache_dir() -> Optional[Path]:
    raw = os.environ.get("EMBEDDINGGEMMA_EMBED_CACHE_DIR", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def load_cached_embedding(text: str) -> Optional[np.ndarray]:
    cache_dir = embedding_cache_dir()
    if cache_dir is None:
        return None
    path = cache_dir / f"{cache_key(text)}.npy"
    if not path.exists():
        return None
    emb = np.load(path)
    if emb.shape != (emb.shape[0],) or emb.dtype != np.float32:
        return None
    return emb


def save_cached_embedding(text: str, emb: np.ndarray) -> None:
    cache_dir = embedding_cache_dir()
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(text)}.npy"
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, emb.astype(np.float32))
    tmp.replace(path)


def write_inputs(ids: list[int], embeds_output: Path, bias_output: Path) -> None:
    table = token_table()
    embeds = np.asarray(table[np.asarray(ids, dtype=np.int64)], dtype=np.float32) * math.sqrt(HIDDEN)
    embeds.reshape(1, SEQ_LEN, HIDDEN).tofile(embeds_output)

    ids_array = np.asarray(ids, dtype=np.int32)
    bias = np.where(ids_array == PAD_ID, MASK_BIAS, 0.0).astype(np.float32)
    bias.reshape(1, 1, 1, SEQ_LEN).tofile(bias_output)


def official_tail(hidden: np.ndarray, ids: list[int]) -> np.ndarray:
    mask = (np.asarray(ids, dtype=np.int32) != PAD_ID).astype(np.float32)
    pooled = (hidden.reshape(SEQ_LEN, HIDDEN) * mask[:, None]).sum(axis=0) / max(float(mask.sum()), 1.0)
    w2, w3 = dense_tail_weights()
    out = pooled @ w2.T @ w3.T
    out = out.astype(np.float32)
    out /= max(float(np.linalg.norm(out)), 1e-12)
    return out


def embed_text(text: str, work_dir: Path, stem: str = "query", verbose: bool = False) -> np.ndarray:
    timing = os.environ.get("EMBEDDINGGEMMA_TIMING", "0") == "1"
    t_start = time.perf_counter()
    cached = load_cached_embedding(text)
    if cached is not None:
        if timing:
            print(f"embed_timing cache_hit=1 total_s={time.perf_counter() - t_start:.6f}", file=sys.stderr, flush=True)
        return cached

    work_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ids = token_ids(text)
    token_s = time.perf_counter() - t0
    embeds_path = work_dir / f"{stem}_embeds_f32.dat"
    bias_path = work_dir / f"{stem}_attention_bias_f32.dat"
    raw_output = work_dir / f"{stem}_hidden_f32.dat"
    sample = work_dir / f"{stem}_sample.txt"

    t0 = time.perf_counter()
    write_inputs(ids, embeds_path, bias_path)
    inputs_s = time.perf_counter() - t0
    sample.write_text(
        "[network]\n"
        f"{ROOT / 'network_binary.nb'}\n"
        "[input]\n"
        f"{embeds_path}\n"
        f"{bias_path}\n"
        "[output]\n"
        f"{raw_output}\n",
        encoding="ascii",
    )

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = VIPLITE_LIB + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    stdout = None if verbose else subprocess.DEVNULL
    t0 = time.perf_counter()
    subprocess.run(
        [VPM_RUN, "-s", str(sample), "-l", "1", "-b", "0"],
        cwd=work_dir,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        check=True,
    )
    npu_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    hidden = np.fromfile(raw_output, dtype=np.float32)
    if hidden.size != SEQ_LEN * HIDDEN:
        raise RuntimeError(f"expected {SEQ_LEN * HIDDEN} hidden values, got {hidden.size}")
    if not np.isfinite(hidden).all():
        raise RuntimeError("NPU returned non-finite hidden states")
    out = official_tail(hidden, ids)
    tail_s = time.perf_counter() - t0
    save_cached_embedding(text, out)
    if timing:
        print(
            "embed_timing "
            f"cache_hit=0 token_s={token_s:.6f} inputs_s={inputs_s:.6f} "
            f"npu_s={npu_s:.6f} tail_s={tail_s:.6f} total_s={time.perf_counter() - t_start:.6f}",
            file=sys.stderr,
            flush=True,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="does Cubie A7S support NVMe storage?")
    parser.add_argument("--verbose-npu", action="store_true")
    parser.add_argument("--cache-dir", help="Optional persistent embedding cache directory")
    parser.add_argument("--timing", action="store_true", help="Print embedding timing breakdown to stderr")
    args = parser.parse_args()
    if args.cache_dir:
        os.environ["EMBEDDINGGEMMA_EMBED_CACHE_DIR"] = args.cache_dir
    if args.timing:
        os.environ["EMBEDDINGGEMMA_TIMING"] = "1"
    emb = embed_text(args.text, ROOT / "work", "text", args.verbose_npu)
    out = ROOT / "embedding_text_bias_hidden_tail_f32.dat"
    emb.astype(np.float32).tofile(out)
    print(f"text={args.text!r}")
    print(f"embedding_file={out}")
    print(f"embedding_norm={float(np.linalg.norm(emb)):.8g}")
    print("embedding_first16=" + " ".join(f"{v:.8g}" for v in emb[:16]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
