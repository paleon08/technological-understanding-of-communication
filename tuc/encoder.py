# tuc/encoder.py
from __future__ import annotations
import hashlib, numpy as np

DIM = 512

def _l2n(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)

def _hash_embed(text: str, dim: int = DIM) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    return _l2n(rng.standard_normal(dim))

def encode_text(texts: list[str]) -> np.ndarray:
    """지금은 hash 기반 임베더. 나중에 CLAP로 이 함수만 교체."""
    return np.vstack([_hash_embed(t) for t in texts]).astype("float32")
