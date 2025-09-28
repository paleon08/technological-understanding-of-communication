import sys, pathlib, csv
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from pathlib import Path
import numpy as np
from lib.features.w2v2 import Wav2Vec2Embedder

IN_DIR  = Path("data/clips")
OUT_DIR = Path("data/embeddings/w2v2")
INDEX_CSV = OUT_DIR / "index.csv"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    embedder = Wav2Vec2Embedder()
    rows = []
    for wav in sorted(IN_DIR.glob("*.wav")):
        emb = embedder(wav)  # [C]
        np.save(OUT_DIR / (wav.stem + ".npy"), emb)
        rows.append([wav.name, str(OUT_DIR / (wav.stem + ".npy")), emb.shape[0]])
    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file","npy","dim"]); w.writerows(rows)
    print(f"[OK] saved {len(rows)} embeddings -> {OUT_DIR}")

if __name__ == "__main__":
    main()
