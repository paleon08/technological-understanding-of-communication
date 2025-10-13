# tuc/encoder.py  — 텍스트 전용 간단 래퍼
from __future__ import annotations
import os, numpy as np, torch
from transformers import AutoTokenizer, AutoModel

_DEFAULT_TEXT = os.getenv("TUC_TEXT_MODEL", "intfloat/e5-base-v2")
_REQUIRE_TEXT = os.getenv("TUC_REQUIRE_TEXT", "1") == "1"  # 기본값: 텍스트 모델 필수

_tok = _mdl = None
_dev = "cuda" if torch.cuda.is_available() else "cpu"

def _load():
    global _tok, _mdl
    if _mdl is not None: return
    try:
        _tok = AutoTokenizer.from_pretrained(_DEFAULT_TEXT)
        _mdl = AutoModel.from_pretrained(_DEFAULT_TEXT).to(_dev).eval()
        print(f"[tuc.encoder] text-model={_DEFAULT_TEXT} device={_dev}")
    except Exception as e:
        if __REQUIRE_TEXT:
            raise RuntimeError(f"Text model required but failed: {e}")
        raise

@torch.no_grad()
def encode_text(texts: list[str]) -> np.ndarray:
    _load()
    toks = _tok(texts, padding=True, truncation=True, return_tensors="pt").to(_dev)
    out  = _mdl(**toks)
    last = getattr(out, "last_hidden_state", None)
    if last is None:
        raise RuntimeError("Model has no last_hidden_state; add model-specific pooling.")
    vec  = last.mean(dim=1)                    # [B, D]
    vec  = torch.nn.functional.normalize(vec, dim=-1)
    return vec.detach().cpu().numpy().astype(np.float32)
