from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import os
import numpy as np

try:
    import soundfile as sf  # 경량 오디오 로더
except Exception:
    sf = None

try:
    import cv2  # 선택: 비디오 프레임 샘플링
except Exception:
    cv2 = None

ArrayLike = Union[List[float], np.ndarray]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n


@dataclass
class InputPayload:
    """원시 입력 + 메타데이터 컨테이너"""
    kind: str                         # 'text' | 'audio' | 'video' | 'sensor'
    data: Any                         # 문자열/np.ndarray/파일경로
    rate: Optional[int] = None        # 오디오/센서 샘플레이트
    channels: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


class Ingestor:
    """다양한 입력을 feature(z)로 통일하는 레이어.
    여기서는 가벼운 통계/샘플링만 수행하고, 의미 임베딩은 encoder가 담당.
    """

    def from_text(self, text: str, meta: Optional[Dict[str, Any]] = None) -> InputPayload:
        assert isinstance(text, str) and len(text.strip()) > 0, "빈 텍스트"
        return InputPayload(kind='text', data=text.strip(), meta=meta or {})

    def from_audio_file(self, path: str, target_rate: int = 16000) -> InputPayload:
        assert os.path.isfile(path), f"오디오 파일 없음: {path}"
        if sf is None:
            raise RuntimeError("soundfile 미설치: pip install soundfile")
        wav, sr = sf.read(path, dtype='float32', always_2d=True)
        # 모노 변환
        wav = wav.mean(axis=1)
        # 간단 리샘플(최근접; torchaudio 없을 때 임시)
        if sr != target_rate:
            ratio = target_rate / sr
            idx = (np.arange(int(len(wav) * ratio)) / ratio).astype(np.int64)
            idx = np.clip(idx, 0, len(wav) - 1)
            wav = wav[idx]
            sr = target_rate
        return InputPayload(kind='audio', data=wav, rate=sr, channels=1, meta={'path': path})

    def from_video_file(self, path: str, fps: int = 2, max_frames: int = 32) -> InputPayload:
        assert os.path.isfile(path), f"비디오 파일 없음: {path}"
        if cv2 is None:
            raise RuntimeError("opencv-python 미설치: pip install opencv-python")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"비디오 열기 실패: {path}")
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(round(input_fps / max(1, fps))), 1)
        frames = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                # 224x224 중앙크롭 + 리사이즈(간단)
                h, w = frame.shape[:2]
                m = min(h, w)
                y0 = (h - m) // 2
                x0 = (w - m) // 2
                crop = frame[y0:y0+m, x0:x0+m]
                crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                crop = crop.astype(np.float32) / 255.0
                frames.append(crop)
                if len(frames) >= max_frames:
                    break
            i += 1
        cap.release()
        if len(frames) == 0:
            raise RuntimeError("프레임 추출 실패")
        arr = np.stack(frames, axis=0)  # [T, 224, 224, 3]
        return InputPayload(kind='video', data=arr, meta={'path': path, 'stride': step})

    def from_sensor_array(self, x: ArrayLike, rate: Optional[int] = None, meta: Optional[Dict[str, Any]] = None) -> InputPayload:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]  # [N,1]
        return InputPayload(kind='sensor', data=arr, rate=rate, meta=meta or {})
