# tuc/loading_base_model/encoder.py — E5 + prefix + robust subspace loader (R^d projection)
from __future__ import annotations
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ---- ENV ----
_MODEL = os.getenv("TUC_TEXT_MODEL", "intfloat/e5-base-v2")
_DEVICE = os.getenv("TUC_DEVICE", "auto")            # auto|cpu|cuda
_DTYPE  = os.getenv("TUC_DTYPE", "auto")             # auto|float32|float16|bfloat16
_NORMALIZE = os.getenv("TUC_NORMALIZE", "1") == "1"  # L2 normalize at end
_E5_MODE   = os.getenv("TUC_E5_MODE", "passage")     # query|passage

_USE_SUB   = os.getenv("TUC_USE_SUBSPACE", "0") == "1"
_SUB_PATH  = os.getenv("TUC_SUBSPACE_PATH", "artifacts/subspace/canon_subspace.npz")

_tok = None
_mdl = None
_dev: Optional[torch.device] = None
_torch_dtype: Optional[torch.dtype] = None

# Loaded subspace cache
_SUBSPACE: Optional[dict] = None   # keys: mean[d], basis[d,k], center(bool), k(int)

def _pick_device(dev: str) -> torch.device:
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

def _pick_dtype(dt: str) -> torch.dtype:
    if dt == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dt]

def _load_subspace() -> Optional[dict]:
    """
    Accept both schemas:
    (A) mean/basis[=R^{d x k}]             (common)
    (B) mu/B(+center) with B orthonormal   (your earlier projector)
    Always project back to R^d via P = B B^T (mean-centered if center=True).
    """
    if not _USE_SUB or not os.path.exists(_SUB_PATH):
        return None
    data = np.load(_SUB_PATH, allow_pickle=False)
    if "basis" in data and "mean" in data:
        basis = data["basis"].astype("float32")
        mean  = data["mean"].astype("float32")
        k = int(data["k"]) if "k" in data else basis.shape[1]
        center = True if "center" not in data else bool(int(np.array(data["center"]).ravel()[0]))
    elif "B" in data and "mu" in data:
        basis = data["B"].astype("float32")
        mean  = data["mu"].astype("float32")
        k = basis.shape[1]
        center = True
        if "center" in data:
            try:
                center = bool(int(np.array(data["center"]).ravel()[0]))
            except Exception:
                center = True
    else:
        raise RuntimeError(f"Unsupported subspace npz keys in {_SUB_PATH}; expected (mean,basis) or (mu,B).")
    mean = mean.reshape(1, -1).astype("float32")
    return {"basis": basis, "mean": mean, "k": k, "center": center}

def _project_Rd(X: np.ndarray, sub: dict) -> np.ndarray:
    """
    Project to span(B) then return to R^d:  X' = mu + (X - mu) @ (B @ B^T).
    Keeps original dimension d for full pipeline compatibility.
    """
    B = sub["basis"]          # [d, k]
    mu = sub["mean"]          # [1, d]
    center = sub["center"]
    X2 = np.atleast_2d(X).astype("float32")
    Xc = X2 - mu if center else X2
    # P = B B^T  (d x d) applied as (Xc @ (B @ B^T)) to avoid forming P explicitly:
    Xp = Xc @ (B @ B.T)       # [n, d]
    Y  = Xp + (mu if center else 0.0)
    return Y if X.ndim == 2 else Y[0]

def _load():
    global _tok, _mdl, _dev, _torch_dtype, _SUBSPACE
    if _mdl is not None:
        return
    _dev = _pick_device(_DEVICE)
    _torch_dtype = _pick_dtype(_DTYPE)
    _tok = AutoTokenizer.from_pretrained(_MODEL, trust_remote_code=True)
    _mdl = AutoModel.from_pretrained(_MODEL, torch_dtype=_torch_dtype, trust_remote_code=True)
    _mdl.to(_dev).eval()
    _SUBSPACE = _load_subspace()
    msg = f"[tuc.encoder] model={_MODEL} device={_dev} dtype={_torch_dtype}"
    if _SUBSPACE is not None:
        msg += f" | subspace(k={_SUBSPACE['k']}, center={_SUBSPACE['center']})"
    print(msg)

def _prefix(texts: List[str], mode: Optional[str]) -> List[str]:
    m = (mode or _E5_MODE).strip().lower()
    if m not in ("query", "passage"):
        m = "passage"
    return [f"{m}: {t}" for t in texts]

@torch.no_grad()
def encode_text(texts: List[str], mode: Optional[str] = None) -> np.ndarray:
    """
    E5 with prefix + CLS pooling + (optional) subspace projection back to R^d + L2 normalize.
    """
    _load()
    toks = _tok(_prefix(texts, mode), padding=True, truncation=True, return_tensors="pt", max_length=512)
    toks = {k: v.to(_dev) for k, v in toks.items()}
    out  = _mdl(**toks)
    last = getattr(out, "last_hidden_state", None)
    if last is None:
        raise RuntimeError("Model has no last_hidden_state; add pooling.")
    vec  = last[:, 0].detach().cpu().float().numpy()  # CLS
    if _SUBSPACE is not None:
        vec = _project_Rd(vec, _SUBSPACE)
    if _NORMALIZE:
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    return vec.astype("float32")
