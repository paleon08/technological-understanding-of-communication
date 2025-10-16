# tuc/loading_base_model/encoder.py — E5 + prefix + optional subspace
from __future__ import annotations
import os
from typing import List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ---- 환경 변수 기본값 ----
_MODEL = os.getenv("TUC_TEXT_MODEL", "intfloat/e5-base-v2")
_DEVICE = os.getenv("TUC_DEVICE", "auto")            # auto|cpu|cuda
_DTYPE  = os.getenv("TUC_DTYPE", "auto")             # auto|float32|float16|bfloat16
_NORMALIZE = os.getenv("TUC_NORMALIZE", "1") == "1"  # L2 정규화
_E5_MODE   = os.getenv("TUC_E5_MODE", "passage")     # query|passage

# (옵션) 하위공간
_USE_SUB = os.getenv("TUC_USE_SUBSPACE", "0") == "1"
_SUB_PATH = os.getenv("TUC_SUBSPACE_PATH", "artifacts/subspace/canon_subspace.npz")

_tok = None
_mdl = None
_dev: Optional[torch.device] = None
_torch_dtype: Optional[torch.dtype] = None
_subspace = None  # {"mean": [1,d], "basis":[d,k], "k":int, "whiten":bool}

def _pick_device(dev: str) -> torch.device:
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

def _pick_dtype(dt: str) -> torch.dtype:
    if dt == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dt]

def _load_sub():
    global _subspace
    if not _USE_SUB: 
        return
    if os.path.exists(_SUB_PATH):
        data = np.load(_SUB_PATH, allow_pickle=False)
        _subspace = {
            "mean": data["mean"].astype("float32"),
            "basis": data["basis"].astype("float32"),
            "k": int(data.get("k", data["basis"].shape[1])),
            "whiten": bool(int(data.get("whiten", np.array([0]))[0])) if "whiten" in data else False,
        }

def _apply_sub(vecs: np.ndarray) -> np.ndarray:
    """[n,d] -> [n,k] 로 투사. 앵커/질의 모두 encode_text를 쓰므로 일관."""
    if _subspace is None: 
        return vecs
    mu = _subspace["mean"]  # [1,d]
    W  = _subspace["basis"] # [d,k]
    Z = (vecs - mu) @ W     # [n,k]
    # 투사 후 L2 정규화는 아래 공통 루틴에서 처리
    return Z.astype("float32")

def _load():
    global _tok, _mdl, _dev, _torch_dtype
    if _mdl is not None: 
        return
    _dev = _pick_device(_DEVICE)
    _torch_dtype = _pick_dtype(_DTYPE)
    _tok = AutoTokenizer.from_pretrained(_MODEL, trust_remote_code=True)
    _mdl = AutoModel.from_pretrained(_MODEL, torch_dtype=_torch_dtype, trust_remote_code=True)
    _mdl.to(_dev).eval()
    _load_sub()
    msg = f"[tuc.encoder] model={_MODEL} device={_dev} dtype={_torch_dtype}"
    if _subspace is not None:
        msg += f" | subspace(k={_subspace['k']})"
    print(msg)

def _prefix(texts: List[str], mode: Optional[str]) -> List[str]:
    m = (mode or _E5_MODE).strip().lower()
    if m not in ("query","passage"): 
        m = "passage"
    return [f"{m}: {t}" for t in texts]

@torch.no_grad()
def encode_text(texts: List[str], mode: Optional[str] = None) -> np.ndarray:
    """
    texts: 관측 텍스트(표층 사실만; 의미 단어 금지)
    mode : "query" | "passage" (E5 프리픽스)
    return: np.ndarray [B, D] 또는 [B, k] (하위공간 on 시)
    """
    _load()
    toks = _tok(_prefix(texts, mode), padding=True, truncation=True, return_tensors="pt", max_length=512)
    toks = {k: v.to(_dev) for k, v in toks.items()}
    out  = _mdl(**toks)
    last = getattr(out, "last_hidden_state", None)
    if last is None:
        raise RuntimeError("Model has no last_hidden_state; add pooling.")
    vec  = last[:, 0].detach().cpu().float().numpy()   # [CLS] 풀링
    vec  = _apply_sub(vec)                             # (옵션) 하위공간 정사영
    if _NORMALIZE:
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    return vec.astype("float32")
