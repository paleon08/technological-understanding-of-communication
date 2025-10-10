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
# --- 기존 encode_text 아래에 추가 ---
from .alignment import apply_alignment

class Projector:
    """
    통합 프로젝터:
    - text(list[str]) -> encode_text
    - audio(paths)    -> Wav2Vec2Embedder로 임베딩 후 (선택) 정렬행렬 W 적용
    - apply(vecs)     -> 정렬행렬 W 적용만
    """
    def __init__(self, W: np.ndarray | None = None):
        self.W = W

    @classmethod
    def from_alignment(cls, W_path: str | Path):
        import numpy as np
        W = np.load(W_path).astype("float32")
        return cls(W)

    def text(self, texts: list[str]) -> np.ndarray:
        V = encode_text(texts)
        return V

    def audio(self, paths: list[str | Path]) -> np.ndarray:
        try:
            from tuc.ops.features.w2v2 import Wav2Vec2Embedder
        except ModuleNotFoundError:
            from .ops.features.w2v2 import Wav2Vec2Embedder  # 레거시 폴백
        emb = Wav2Vec2Embedder()
        vecs = [emb(p) for p in paths]
        V = np.stack(vecs,0).astype("float32")
        return self.apply(V)

    def apply(self, V: np.ndarray) -> np.ndarray:
        if self.W is None: return V
        return apply_alignment(V, self.W).astype("float32")
