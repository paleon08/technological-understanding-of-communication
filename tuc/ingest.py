# tuc/ingest.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os, json
import numpy as np
import soundfile as sf

from tuc.audio_backend import get as get_audio_backend

@dataclass
class EmbedResult:
    key: str
    path: str
    out_dir: str
    meta: Dict[str, Any]

def _save_vec(out_dir: Path, key: str, vec: np.ndarray, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{key}.npy", vec.astype(np.float32))
    with open(out_dir / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def embed_audio_file(path: str, out_dir: str,
                     backend: str | None = None,
                     model_id: str | None = None) -> EmbedResult:
    """
    path: 입력 오디오 파일
    out_dir: 저장 폴더
    backend: None|'external'|'wav2vec2' (None이면 external)
    model_id: 외부 모델 ID(옵션, 없으면 내부 기본값)
    """
    p = Path(path)
    wav, sr = sf.read(p)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    be = get_audio_backend(backend, model_id=model_id)
    vec = be.encode_wave(wav, int(sr))  # [D]

    key = p.stem
    meta = {
        "path": str(p),
        "backend": be.name,
        "sr": int(sr),
        "dim": int(vec.shape[-1]),
        "model_id": getattr(be, "model_id", None),
    }
    _save_vec(Path(out_dir), key, vec, meta)
    return EmbedResult(key=key, path=str(p), out_dir=str(out_dir), meta=meta)

def embed_from_folder(in_root: str, out_root: str,
                      backend: str | None = None,
                      model_id: str | None = None,
                      exts=(".wav",".mp3",".flac",".ogg",".m4a")) -> list[EmbedResult]:
    in_root = Path(in_root)
    out_root = Path(out_root)
    files = [p for p in in_root.rglob("*") if p.suffix.lower() in exts]
    results: list[EmbedResult] = []
    for p in files:
        try:
            r = embed_audio_file(str(p), str(out_root), backend=backend, model_id=model_id)
            results.append(r)
        except Exception as e:
            print("[ingest] fail:", p, e)
    return results
