#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", help="Path to an F16/F32 GGUF model")
    parser.add_argument("tensor", help="Tensor name, for example blk.0.attn_k.weight")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--gguf-py", default=None, help="Path to llama.cpp/gguf-py if gguf is not installed")
    args = parser.parse_args()

    if args.gguf_py:
        sys.path.insert(0, args.gguf_py)

    from gguf import GGUFReader

    os.makedirs(args.out_dir, exist_ok=True)
    reader = GGUFReader(args.gguf)

    selected = None
    for tensor in reader.tensors:
        if tensor.name == args.tensor:
            selected = tensor
            break

    if selected is None:
        raise SystemExit(f"tensor not found: {args.tensor}")

    data = np.asarray(selected.data)
    if data.ndim != 2:
        raise SystemExit(f"expected a 2D tensor, got shape {data.shape}")

    stem = args.tensor.replace(".", "_")
    if data.dtype == np.float16:
        mode = "f16a"
        weight = np.asarray(data, dtype=np.float16)
        weight_path = os.path.join(args.out_dir, f"{stem}_f16.bin")
    elif data.dtype == np.float32:
        mode = "f32"
        weight = np.asarray(data, dtype=np.float32)
        weight_path = os.path.join(args.out_dir, f"{stem}_f32.bin")
    else:
        raise SystemExit(f"unsupported tensor dtype: {data.dtype}")

    rows, cols = weight.shape
    vec = ((((np.arange(cols, dtype=np.int32) * 19) % 2001) - 1000).astype(np.float32) / 1000.0)
    vec_path = os.path.join(args.out_dir, f"{stem}_vec_f32.bin")

    weight.tofile(weight_path)
    vec.tofile(vec_path)

    print(f"mode={mode}")
    print(f"rows={rows}")
    print(f"cols={cols}")
    print(f"weight={weight_path}")
    print(f"vector={vec_path}")
    print("command:")
    print(f"  ./run_matvec_repro.sh {rows} {cols} {mode} {weight_path} {vec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
