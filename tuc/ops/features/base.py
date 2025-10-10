from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class BaseAudioEmbedder(ABC):
    @abstractmethod
    def __call__(self, wav_path: Path) -> np.ndarray:
        ...
