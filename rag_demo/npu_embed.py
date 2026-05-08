#!/usr/bin/env python3
import sys
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from embed_text_bias_hidden_npu import embed_text  # noqa: E402,F401
