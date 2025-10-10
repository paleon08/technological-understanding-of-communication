import numpy as _np
from typing import List as _List, Optional as _Optional, Dict as _Dict, Any as _Any

try:
    # 예: sentence-transformers 또는 프로젝트의 텍스트 인코더(실환경에 맞게 교체)
    from sentence_transformers import SentenceTransformer as _Sbert
    _SBERT = _Sbert('all-MiniLM-L6-v2')
except Exception:
    _SBERT = None


def _l2_normalize(x: _np.ndarray, eps: float = 1e-12) -> _np.ndarray:
    n = _np.linalg.norm(x) + eps
    return x / n


def _safe_text_embed(texts: _List[str]) -> _np.ndarray:
    """경량 텍스트 임베딩(임시). 실제에선 CLAP/프로젝트 내 텍스트 인코더로 교체.
    의존성 없으면 bag-of-char 평균으로 더미 벡터 생성(형상 고정 256).
    """
    if _SBERT is not None:
        emb = _SBERT.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(_np.float32)
    # fallback: char-bag dummy
    dim = 256
    out = _np.zeros((len(texts), dim), dtype=_np.float32)
    for i, t in enumerate(texts):
        v = _np.zeros(dim, dtype=_np.float32)
        for ch in t:
            v[ord(ch) % dim] += 1.0
        out[i] = v / (_np.linalg.norm(v) + 1e-12)
    return out


class Projector:
    """z → q 로 투영(정렬 없음 버전). configs/projection.yml 있으면 불러 사용.
    - text: z 는 임베딩 그 자체
    - audio: z 는 파형 통계 → 작은 MLP/선형층 대체로 평균+표준편차를 concat 하여 투영
    - video: z 는 [T,224,224,3] → 프레임 평균/분산 + 모션 차분
    - sensor: z 는 [N,C] → 채널별 통계
    실제 정렬은 나중에 W,b 로 교체/추가.
    """
    def __init__(self, proj_cfg: _Optional[_Dict[str, _Any]] = None):
        self.cfg = proj_cfg or {}

    def text(self, payload) -> _np.ndarray:
        z = _safe_text_embed([payload.data])[0]
        return z.astype(_np.float32)

    def audio(self, payload) -> _np.ndarray:
        x = _np.asarray(payload.data, dtype=_np.float32)
        mean = x.mean()
        std = x.std() + 1e-8
        maxv = float(_np.max(_np.abs(x)) + 1e-8)
        # 간단한 스펙트럼 에너지(DFT 일부)
        n_fft = min(len(x), 4096)
        spec = _np.fft.rfft(x[:n_fft])
        pwr = _np.abs(spec)
        bands = _np.array_split(pwr, 8)
        band_stats = _np.array([b.mean() for b in bands], dtype=_np.float32)
        z = _np.concatenate([[mean, std, maxv], band_stats], axis=0)  # dim=11
        return _l2_normalize(z.astype(_np.float32))

    def video(self, payload) -> _np.ndarray:
        arr = _np.asarray(payload.data, dtype=_np.float32)  # [T,H,W,3]
        mean = arr.mean(axis=(0, 1, 2))            # [3]
        std = arr.std(axis=(0, 1, 2)) + 1e-8       # [3]
        # 간단한 모션 힌트: 연속 프레임 차 평균
        diffs = _np.mean(_np.abs(_np.diff(arr, axis=0)), axis=(0, 1, 2)) if arr.shape[0] > 1 else _np.zeros(3, _np.float32)
        z = _np.concatenate([mean, std, diffs], axis=0)  # dim=9
        return _l2_normalize(z.astype(_np.float32))

    def sensor(self, payload) -> _np.ndarray:
        x = _np.asarray(payload.data, dtype=_np.float32)  # [N,C]
        mean = x.mean(axis=0)
        std = x.std(axis=0) + 1e-8
        z = _np.concatenate([mean, std], axis=0)
        return _l2_normalize(z.astype(_np.float32))
