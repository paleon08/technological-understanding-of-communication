from pathlib import Path
import torch, torchaudio, numpy as np
from .base import BaseAudioEmbedder

class Wav2Vec2Embedder(BaseAudioEmbedder):
    def __init__(self, device=None, layer: int = -1):
        self.bundle = torchaudio.pipelines.WAV2VEC2_BASE  # 16kHz
        self.model = self.bundle.get_model().eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.layer = layer  # 마지막 히든레이어

    def __call__(self, wav_path: Path) -> np.ndarray:
        wav, sr = torchaudio.load(str(wav_path))
        if sr != self.bundle.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.bundle.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)  # mono
        with torch.no_grad():
            x = wav.to(self.device)
            feats, _ = self.model.extract_features(x)  # list of [B,T,C]
            h = feats[self.layer].squeeze(0)          # [T,C]
            emb = h.mean(dim=0).cpu().numpy()         # time-average -> [C]
        return emb
