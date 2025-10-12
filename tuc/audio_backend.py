# tuc/audio_backend.py
from __future__ import annotations
import os
import numpy as np

class BaseAudioBackend:
    name: str = "base"
    def encode_wave(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """wav: float32 mono [-1,1], shape [T] → return: float32 [D] L2-normalized"""
        raise NotImplementedError

# 1) 기존 Wav2Vec2 경로(이미 프로젝트에 있다면 여기서 호출)
class Wav2Vec2Backend(BaseAudioBackend):
    name = "wav2vec2"
    def __init__(self):
        # 필요 시 여기서 기존 모듈 import (예: lib.ops.features.w2v2)
        try:
            from lib.ops.features import w2v2 as _w2v2  # 프로젝트 구조에 맞춰 조정
            self._impl = _w2v2
        except Exception as e:
            self._impl = None
            print("[Wav2Vec2Backend] fallback: impl not found:", e)

    def encode_wave(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if self._impl is None:
            # 아주 단순 폴백(안전망): 스펙트럼 파워 요약 (임시)
            x = np.abs(np.fft.rfft(wav.astype(np.float32)))
            x = x / (np.linalg.norm(x) + 1e-8)
            return x.astype(np.float32)
        # 프로젝트의 w2v2 추출 함수에 맞춰 호출 (아래는 예시 형태)
        # vec = self._impl.embed_one(wav, sr)  # 예: [D]
        # return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)
        # ↑ 위 함수명이 다르면 실제 프로젝트 함수명에 맞춰 수정
        raise NotImplementedError("Wav2Vec2Backend: 프로젝트의 w2v2 임베딩 호출부를 연결하세요.")

# 2) 외부 모델(Transformers) 공용 래퍼
class HFExternalAudioBackend(BaseAudioBackend):
    """
    HuggingFace 오디오 모델 공용 래퍼
    환경변수 TUC_AUDIO_MODEL 로 모델 ID 지정 (예: 'laion/clap-htsat-fused' 등)
    - get_audio_features 가 있으면 그걸 사용
    - 없으면 last_hidden_state 를 평균 풀링
    """
    name = "external"

    def __init__(self, model_id: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoProcessor, AutoModel

        self.model_id = model_id or os.getenv("TUC_AUDIO_MODEL", "laion/clap-htsat-fused")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Processor/Model 로드
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

        # 사용할 전방 함수 선택
        self._use_get_audio_features = hasattr(self.model, "get_audio_features")

    def encode_wave(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import torch
        # mono 보장
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)

        # 일부 모델은 샘플레이트 고정(예: 32k, 48k)이 필요할 수 있음 → 필요 시 리샘플
        # 여기선 processor가 내부에서 처리하거나, 안 되면 그대로 시도
        inputs = self.processor(audios=wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self._use_get_audio_features:
                feats = self.model.get_audio_features(**inputs)  # [1, D]
            else:
                out = self.model(**inputs)
                if hasattr(out, "last_hidden_state"):
                    feats = out.last_hidden_state.mean(dim=1)  # [1, D]
                else:
                    # 안전망: 모델에 따라 출력 키가 다를 수 있음
                    # 가장 큰 텐서를 찾아 평균 풀링
                    tensors = [v for v in out.__dict__.values() if hasattr(v, "dim")]
                    if not tensors:
                        raise RuntimeError("No tensor outputs from external audio model.")
                    t = max(tensors, key=lambda x: x.numel())
                    if t.dim() >= 2:
                        feats = t.mean(dim=1)
                    else:
                        feats = t.unsqueeze(0)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats[0].detach().cpu().numpy().astype(np.float32)

def get(name: str) -> BaseAudioBackend:
    name = (name or "").lower()
    if name in {"wav2vec2", "w2v2"}:
        return Wav2Vec2Backend()
    if name in {"naturelm", "external", "hf"}:
        return HFExternalAudioBackend()
    raise ValueError(f"Unknown audio backend: {name}")
