# scripts/embed_from_folder.py
from pathlib import Path
import argparse, numpy as np, pandas as pd
from tuc.ops.features.w2v2 import Wav2Vec2Embedder

def embed_dir(in_dir: Path, out_dir: Path):
    wavs = sorted([p for p in in_dir.rglob("*") if p.suffix.lower() in [".wav",".mp3",".flac",".ogg",".m4a"]])
    if not wavs:
        print(f"[skip] no audio: {in_dir}"); return
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_fn = out_dir / f"{in_dir.name}.npy"
    csv_fn = out_dir / f"{in_dir.name}.csv"

    embedder = Wav2Vec2Embedder()
    rows, embs = [], []
    for i, p in enumerate(wavs):
        try:
            v = embedder(p)                  # <-- 네 기존 임베더 재사용
            embs.append(v)
            rows.append({"index": i, "file": str(p).replace("\\","/")})
        except Exception as e:
            print(f"[warn] {p}: {e}")

    if embs:
        np.save(emb_fn, np.stack(embs))
        pd.DataFrame(rows).to_csv(csv_fn, index=False)
        print(f"[done] {in_dir.name}: {len(embs)} -> {emb_fn}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="input folder, e.g., data/crestedgecko_raw")
    ap.add_argument("--out", dest="out_dir", required=True, help="output folder, e.g., artifacts/audio_embeds/crestedgecko")
    args = ap.parse_args()
    embed_dir(Path(args.in_dir), Path(args.out_dir))
