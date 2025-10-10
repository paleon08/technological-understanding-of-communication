# tuc/transforms/audio_w2v2.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from lib.ops.features.w2v2 import Wav2Vec2Embedder  # 기존 구현 재사용
from .base import BaseTransform

class AudioW2V2Transform(BaseTransform):
    def __init__(self, device: str | None = None, layer: int = -1):
        self._embedder = Wav2Vec2Embedder(device=device, layer=layer)
    def __call__(self, wav_path: str | Path) -> np.ndarray:
        return self._embedder(Path(wav_path))
