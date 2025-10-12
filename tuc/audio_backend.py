# tuc/audio_backend.py
from __future__ import annotations
import os
import numpy as np

DEFAULT_MODEL_ID = "laion/clap-htsat-fused"  # 모델 입력이 없을 때 내부 기본값

class BaseAudioBackend:
    name: str = "base"
    def encode_wave(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """wav(float32 mono[-1,1], [T]) -> embedding(float32 [D], L2-normalized)"""
        raise NotImplementedError

# (선택) 프로젝트에 기존 W2V2 임베더가 있다면 여기에 감싸세요.
class Wav2Vec2Backend(BaseAudioBackend):
    name = "wav2vec2"
    def __init__(self):
        try:
            # 프로젝트 내부 구현이 있으면 가져오기
            from tuc.ops.features.w2v2 import Wav2Vec2Embedder
            self._impl = Wav2Vec2Embedder()
        except Exception as e:
            self._impl = None
            print("[Wav2Vec2Backend] impl not found; using trivial fallback:", e)

    def encode_wave(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if self._impl is not None:
            vec = self._impl.embed_one(wav, sr)  # <- 프로젝트 구현에 맞춰 함수명 조정
            v = vec.astype(np.float32)
        else:
            # 아주 간단한 안전망(임시): FFT 파워 요약
            x = np.abs(np.fft.rfft(wav))
            v = x.astype(np.float32)
        # L2 정규화
        n = np.linalg.norm(v) + 1e-8
        return (v / n).astype(np.float32)

class HFExternalAudioBackend(BaseAudioBackend):
    """
    HuggingFace 오디오 모델 공용 래퍼.
    모델 ID 우선순위: 함수 인자 model_id > 환경변수 TUC_AUDIO_MODEL > DEFAULT_MODEL_ID
    - get_audio_features()가 있으면 그걸 사용
    - 없으면 last_hidden_state 평균 풀링
    """
    name = "external"

    def __init__(self, model_id: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoProcessor, AutoModel

        self.model_id = model_id or os.getenv("TUC_AUDIO_MODEL") or DEFAULT_MODEL_ID
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        self._use_get_audio_features = hasattr(self.model, "get_audio_features")

        print(f"[audio_backend] backend=external model_id={self.model_id} device={self.device}")

    def encode_wave(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import torch
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)

        # 일부 모델은 고정 sampling_rate를 요구할 수 있음(필요 시 리샘플 추가)
        inputs = self.processor(audios=wav, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self._use_get_audio_features:
                feats = self.model.get_audio_features(**inputs)  # [1, D]
            else:
                out = self.model(**inputs)
                if hasattr(out, "last_hidden_state"):
                    feats = out.last_hidden_state.mean(dim=1)      # [1, D]
                else:
                    # 가장 큰 텐서를 골라 평균 풀링 (보험)
                    tensors = [v for v in out.__dict__.values() if hasattr(v, "dim")]
                    if not tensors:
                        raise RuntimeError("External audio model returned no tensor outputs.")
                    t = max(tensors, key=lambda x: x.numel())
                    feats = t if t.dim() == 2 else t.unsqueeze(0)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            return feats[0].detach().cpu().numpy().astype(np.float32)

def get(name: str | None, model_id: str | None = None):
    """
    name: 'external' | 'wav2vec2' | None
      - None이면 'external' 사용(모델 ID는 자동 결정)
    """
    name = (name or "external").lower()
    if name in {"external", "hf", "naturelm"}:
        return HFExternalAudioBackend(model_id=model_id)
    if name in {"wav2vec2", "w2v2"}:
        return Wav2Vec2Backend()
    raise ValueError(f"Unknown audio backend: {name}")
