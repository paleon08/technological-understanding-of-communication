from abc import ABC, abstractmethod
import numpy as np
class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, **kwargs) -> np.ndarray: ...
