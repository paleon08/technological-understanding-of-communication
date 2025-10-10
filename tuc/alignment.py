# tuc/alignment.py
from __future__ import annotations
import numpy as np

def l2n(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)

def fit_alignment(B: np.ndarray, T: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
    # B: [N, d_b], T: [N, d_t]
    B = l2n(B.astype("float32")); T = l2n(T.astype("float32"))
    BTB = B.T @ B
    BTt = B.T @ T
    W = np.linalg.solve(BTB + lambda_*np.eye(B.shape[1], dtype=B.dtype), BTt)
    return W.astype("float32")

def apply_alignment(b: np.ndarray, W: np.ndarray) -> np.ndarray:
    v = b.astype("float32").reshape(-1) @ W.astype("float32")
    return l2n(v)
