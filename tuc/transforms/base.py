# tuc/transforms/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, **kwargs) -> np.ndarray:
        """임의의 입력(**kwargs)을 받아 [D] 임베딩 벡터로 변환."""
        raise NotImplementedError
