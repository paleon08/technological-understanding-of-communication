# tuc/ingest/__init__.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Iterable

AUDIO_EXTS = {".wav",".mp3",".flac",".ogg",".m4a"}

@dataclass
class Ingestor:
    """
    표준 Ingestor:
    - process_dir(in_dir, out_dir): 폴더 내 오디오 -> 임베딩 npy & csv
    - iter_audio_files(in_dir): 파일 발견 제너레이터
    """
    embedder: object  # callable(path) -> np.ndarray[D]

    def iter_audio_files(self, in_dir: Path) -> Iterable[Path]:
        for p in sorted(in_dir.rglob("*")):
            if p.suffix.lower() in AUDIO_EXTS:
                yield p

    def process_dir(self, in_dir: Path, out_dir: Path) -> tuple[int, Path, Path]:
        import pandas as pd
        files = list(self.iter_audio_files(in_dir))
        if not files:
            return 0, Path(), Path()
        out_dir.mkdir(parents=True, exist_ok=True)
        npy = out_dir / f"{in_dir.name}.npy"
        csv = out_dir / f"{in_dir.name}.csv"
        embs, rows = [], []
        for p in files:
            try:
                v = self.embedder(p)
                embs.append(v.astype("float32"))
                rows.append({"file": str(p)})
            except Exception as e:
                print(f"[warn] {p}: {e}")
        if embs:
            X = np.stack(embs,0)
            np.save(npy, X)
            pd.DataFrame(rows).to_csv(csv, index=False)
        return len(embs), npy, csv
