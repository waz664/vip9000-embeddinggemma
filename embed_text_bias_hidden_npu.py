#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import sentencepiece as spm


ROOT = Path(__file__).resolve().parent
SEQ_LEN = 128
HIDDEN = 768
VOCAB = 262144
BOS_ID = 2
EOS_ID = 1
PAD_ID = 0
MASK_BIAS = -10000.0


def token_ids(text: str) -> list[int]:
    sp = spm.SentencePieceProcessor(model_file="/home/radxa/embeddinggemma/tokenizer.model")
    pieces = sp.encode(text, out_type=int)
    ids = [BOS_ID] + pieces[: SEQ_LEN - 2] + [EOS_ID]
    ids.extend([PAD_ID] * (SEQ_LEN - len(ids)))
    return ids


def write_inputs(ids: list[int], embeds_output: Path, bias_output: Path) -> None:
    table = np.memmap(ROOT / "token_embedding_fp16.dat", dtype=np.float16, mode="r", shape=(VOCAB, HIDDEN))
    embeds = np.asarray(table[np.asarray(ids, dtype=np.int64)], dtype=np.float32) * math.sqrt(HIDDEN)
    embeds.reshape(1, SEQ_LEN, HIDDEN).tofile(embeds_output)

    ids_array = np.asarray(ids, dtype=np.int32)
    bias = np.where(ids_array == PAD_ID, MASK_BIAS, 0.0).astype(np.float32)
    bias.reshape(1, 1, 1, SEQ_LEN).tofile(bias_output)


def official_tail(hidden: np.ndarray, ids: list[int]) -> np.ndarray:
    mask = (np.asarray(ids, dtype=np.int32) != PAD_ID).astype(np.float32)
    pooled = (hidden.reshape(SEQ_LEN, HIDDEN) * mask[:, None]).sum(axis=0) / max(float(mask.sum()), 1.0)
    w2 = np.load(ROOT / "dense_2_weight_f32.npy")
    w3 = np.load(ROOT / "dense_3_weight_f32.npy")
    out = pooled @ w2.T @ w3.T
    out = out.astype(np.float32)
    out /= max(float(np.linalg.norm(out)), 1e-12)
    return out


def embed_text(text: str, work_dir: Path, stem: str = "query", verbose: bool = False) -> np.ndarray:
    work_dir.mkdir(parents=True, exist_ok=True)
    ids = token_ids(text)
    embeds_path = work_dir / f"{stem}_embeds_f32.dat"
    bias_path = work_dir / f"{stem}_attention_bias_f32.dat"
    raw_output = work_dir / f"{stem}_hidden_f32.dat"
    sample = work_dir / f"{stem}_sample.txt"

    write_inputs(ids, embeds_path, bias_path)
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
    env["LD_LIBRARY_PATH"] = "/home/radxa/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0"
    stdout = None if verbose else subprocess.DEVNULL
    subprocess.run(
        ["/home/radxa/ai-sdk/examples/vpm_run/vpm_run", "-s", str(sample), "-l", "1", "-b", "0"],
        cwd=work_dir,
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        check=True,
    )

    hidden = np.fromfile(raw_output, dtype=np.float32)
    if hidden.size != SEQ_LEN * HIDDEN:
        raise RuntimeError(f"expected {SEQ_LEN * HIDDEN} hidden values, got {hidden.size}")
    if not np.isfinite(hidden).all():
        raise RuntimeError("NPU returned non-finite hidden states")
    return official_tail(hidden, ids)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="does Cubie A7S support NVMe storage?")
    parser.add_argument("--verbose-npu", action="store_true")
    args = parser.parse_args()
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
