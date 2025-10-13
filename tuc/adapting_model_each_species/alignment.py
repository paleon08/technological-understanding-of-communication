# tuc/alignment.py
from __future__ import annotations
import numpy as np

def fit_alignment(B: np.ndarray, T: np.ndarray, lambda_: float = 0.1) -> np.ndarray:
    """
    B: [N, D_b]  behavior vectors
    T: [N, D_t]  text/anchor vectors
    -> W: [D_b, D_t]  with ridge regularization
    """
    assert B.ndim==2 and T.ndim==2 and B.shape[0]==T.shape[0]
    Db, Dt = B.shape[1], T.shape[1]
    BtB = B.T @ B
    W = np.linalg.solve(BtB + lambda_ * np.eye(Db, dtype=B.dtype), B.T @ T)
    return W.astype("float32", copy=False)

def apply_alignment(b: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    b: [D_b] or [N, D_b], W: [D_b, D_t] -> [D_t] or [N, D_t]
    """
    b2 = b.reshape(-1, b.shape[-1])
    v = b2 @ W
    if b.ndim == 1:
        return v[0]
    return v
