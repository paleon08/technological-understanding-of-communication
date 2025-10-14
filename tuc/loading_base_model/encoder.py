# tuc/loading_base_model/encoder.py — E5 계열 텍스트 캐논 로더 + Subspace
from __future__ import annotations
import os
from typing import List, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

try:
    from .subspace import get_active_projector
except Exception:
    get_active_projector = lambda: None  # noqa

# -------- 환경 변수 기본값 --------
_MODEL = os.getenv("TUC_TEXT_MODEL", "intfloat/e5-base-v2")
_DEVICE = os.getenv("TUC_DEVICE", "auto")            # auto|cpu|cuda
_DTYPE  = os.getenv("TUC_DTYPE", "auto")             # auto|float32|float16|bfloat16
_NORMALIZE = os.getenv("TUC_NORMALIZE", "1") == "1"  # L2 정규화 사용 여부
_E5_MODE   = os.getenv("TUC_E5_MODE", "passage")     # query|passage (E5 권장 프리픽스)

_tok = None
_mdl = None
_dev: Optional[torch.device] = None
_torch_dtype: Optional[torch.dtype] = None
_projector = None  # lazy-load

def _pick_device(dev: str) -> torch.device:
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

def _pick_dtype(dt: str) -> torch.dtype:
    if dt == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return mapping[dt]

def _load():
    global _tok, _mdl, _dev, _torch_dtype, _projector
    if _mdl is not None:
        return
    _dev = _pick_device(_DEVICE)
    _torch_dtype = _pick_dtype(_DTYPE)
    _tok = AutoTokenizer.from_pretrained(_MODEL, trust_remote_code=True)
    _mdl = AutoModel.from_pretrained(_MODEL, torch_dtype=_torch_dtype, trust_remote_code=True)
    _mdl.to(_dev).eval()
    _projector = get_active_projector()
    msg = f"[tuc.encoder] model={_MODEL} device={_dev} dtype={_torch_dtype}"
    if _projector is not None:
        msg += f" | subspace(k={_projector.k}, d={_projector.d}, center={_projector.center})"
    print(msg)

def _prefix_e5(texts: List[str], mode: Optional[str]) -> List[str]:
    m = (mode or _E5_MODE).strip().lower()
    if m not in ("query", "passage"):
        m = "passage"
    return [f"{m}: {t}" for t in texts]

@torch.no_grad()
def encode_text(texts: List[str], mode: Optional[str] = None) -> np.ndarray:
    """
    texts: 관측 텍스트(의미 단어 금지; 표층 사실만)
    mode : "query" 또는 "passage" (E5 프리픽스)
    return: np.ndarray [B, D] (하위공간 정사영+L2 정규화까지 반영)
    """
    _load()
    toks = _tok(_prefix_e5(texts, mode), padding=True, truncation=True,
                return_tensors="pt", max_length=512)
    toks = {k: v.to(_dev) for k, v in toks.items()}
    out = _mdl(**toks)
    last = getattr(out, "last_hidden_state", None)
    if last is None:
        raise RuntimeError("Model has no last_hidden_state; add pooling.")
    vec = last[:, 0].detach().cpu().float().numpy()   # [CLS] 풀링 → [B, D]

    # ---- Subspace projection (if active) ----
    if _projector is not None:
        vec = _projector.project_batch(vec)           # [B, D] in span(B)

    # ---- L2 normalize at the end ----
    if _NORMALIZE:
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    return vec.astype(np.float32)
