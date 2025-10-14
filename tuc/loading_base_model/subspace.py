# tuc/loading_base_model/subspace.py
from __future__ import annotations
import os, json, time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

@dataclass
class SubspaceProjector:
    """Orthonormal basis projector: project to span(B) and back to R^d."""
    B: np.ndarray          # [d, k] with orthonormal columns
    mu: np.ndarray         # [d,]
    center: bool           # mean-centering on/off
    meta: Dict[str, Any]   # saved metadata

    @property
    def d(self) -> int: return int(self.B.shape[0])
    @property
    def k(self) -> int: return int(self.B.shape[1])

    def project(self, X: np.ndarray) -> np.ndarray:
        """X: [d] or [n, d] → returns projected in R^d (same dimension)."""
        X2d = np.atleast_2d(X)                       # [n, d]
        Xc  = X2d - self.mu if self.center else X2d
        # P = B B^T (R^d → projection onto span(B))
        Xp  = Xc @ (self.B @ self.B.T)
        Y   = Xp + (self.mu if self.center else 0.0)
        return Y if X.ndim == 2 else Y[0]

    def project_batch(self, X: np.ndarray) -> np.ndarray:
        return self.project(X)

    def save(self, npz_path: str, meta_json_path: Optional[str] = None) -> None:
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, B=self.B.astype(np.float32), mu=self.mu.astype(np.float32),
                 center=np.array([int(self.center)], dtype=np.int32))
        m = dict(self.meta)
        m["k"] = int(self.k); m["d"] = int(self.d)
        m["saved_at"] = int(time.time())
        if meta_json_path is None:
            meta_json_path = os.path.splitext(npz_path)[0] + ".meta.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(npz_path: str) -> "SubspaceProjector":
        data = np.load(npz_path, allow_pickle=False)
        B = data["B"]; mu = data["mu"]; center = bool(int(data["center"][0]))
        meta_json_path = os.path.splitext(npz_path)[0] + ".meta.json"
        meta = {}
        if os.path.exists(meta_json_path):
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        return SubspaceProjector(B=B, mu=mu, center=center, meta=meta)

def _default_path() -> str:
    return os.getenv("TUC_SUBSPACE_PATH", "artifacts/subspace/canon_subspace.npz")

_ACTIVE: Optional[SubspaceProjector] = None

def get_active_projector() -> Optional[SubspaceProjector]:
    global _ACTIVE
    if os.getenv("TUC_USE_SUBSPACE", "1") != "1":
        return None
    if _ACTIVE is not None:
        return _ACTIVE
    path = _default_path()
    if not os.path.exists(path):
        return None
    _ACTIVE = SubspaceProjector.load(path)
    return _ACTIVE

def fit_subspace(anchors: np.ndarray, center: bool = True,
                 var_thresh: float = 0.95, max_dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, float]]:
    """
    anchors: [m, d], L2-normalized rows (E5 vectors).
    center : mean-center before SVD.
    var_thresh: keep top-k s.t. explained variance >= var_thresh.
    max_dim: cap k.
    returns: (B[d,k], mu[d], k, stats)
    """
    assert anchors.ndim == 2, "anchors must be [m, d]"
    m, d = anchors.shape
    mu = anchors.mean(axis=0) if center else np.zeros((d,), dtype=np.float32)
    A = anchors - mu if center else anchors.copy()
    # SVD on [m, d]
    # Note: explained variance proportional to s^2
    U, S, Vt = np.linalg.svd(A, full_matrices=False)  # Vt: [k0, d], k0=min(m,d)
    s2 = S**2
    ratio = s2 / (s2.sum() + 1e-12)
    cum = np.cumsum(ratio)
    k_all = Vt.shape[0]
    k = int(np.searchsorted(cum, var_thresh) + 1)
    if max_dim is not None:
        k = min(k, int(max_dim))
    k = max(1, min(k, k_all))
    B = Vt[:k, :].T         # [d, k]
    stats = {"explained_var": float(cum[k-1]), "k": float(k), "m": float(m), "d": float(d)}
    return B.astype(np.float32), mu.astype(np.float32), k, stats
