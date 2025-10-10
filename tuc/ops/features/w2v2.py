# tuc/ops/features/w2v2.py
from __future__ import annotations
from pathlib import Path
import numpy as np

class Wav2Vec2Embedder:
    """
    Mean-pooled Wav2Vec2 hidden-state -> L2-normalized vector.
    의존: transformers, torch, librosa
    """
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h", device: str | None = None):
        try:
            import torch
            from transformers import AutoProcessor, AutoModel
            import librosa
        except Exception as e:
            raise RuntimeError("Install audio deps: transformers, torch, torchaudio, librosa, soundfile") from e
        self.torch = torch
        self.librosa = librosa
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def __call__(self, audio_path: str | Path) -> np.ndarray:
        y, sr = self.librosa.load(str(audio_path), sr=16000, mono=True)
        import numpy as np
        with self.torch.no_grad():
            inp = self.proc(y, sampling_rate=16000, return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items()}
            out = self.model(**inp).last_hidden_state  # [1, T, D]
            vec = out.mean(dim=1).squeeze(0).detach().cpu().numpy().astype("float32")
        # L2 normalize
        n = np.linalg.norm(vec) + 1e-12
        return (vec / n).astype("float32")
