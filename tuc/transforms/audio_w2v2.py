from pathlib import Path
import numpy as np
from lib.ops.features.w2v2 import Wav2Vec2Embedder
from .base import BaseTransform

class AudioW2V2Transform(BaseTransform):
    def __init__(self, device=None, layer=-1):
        self._embed = Wav2Vec2Embedder(device=device, layer=layer)
    def __call__(self, wav_path: str | Path) -> np.ndarray:
        return self._embed(Path(wav_path))
