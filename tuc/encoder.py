# tuc/encoder.py
from __future__ import annotations
from typing import List
import numpy as np

_SBERT = None
try:
    from sentence_transformers import SentenceTransformer  # optional
    _SBERT = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _SBERT = None

def _l2n(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / n

def encode_text(texts: List[str]) -> np.ndarray:
    """
    Encode list of sentences -> [N,D] L2-normalized float32 matrix.
    Prefers SBERT if available; falls back to deterministic char-bag.
    """
    if _SBERT is not None:
        try:
            E = _SBERT.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            E = E.astype(np.float32, copy=False)
            return E
        except Exception:
            pass

    # Fallback: simple character-bag + L2 norm
    dim = 256
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        v = np.zeros(dim, dtype=np.float32)
        for ch in t:
            v[ord(ch) % dim] += 1.0
        out[i] = _l2n(v)
    return out
