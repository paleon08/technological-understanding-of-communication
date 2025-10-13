# scripts/batch_ingest.py
from pathlib import Path
from tuc.ingest import Ingestor
try:
    from tuc.ops.features.w2v2 import Wav2Vec2Embedder
except ModuleNotFoundError:
    from tuc.ops.features.w2v2 import Wav2Vec2Embedder  # 폴백

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True, help="세션/폴더들이 들어있는 루트")
    ap.add_argument("--out-root", required=True, help="NPY/CSV 저장 루트")
    args = ap.parse_args()

    in_root  = Path(args.in_root)
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    ing = Ingestor(embedder=Wav2Vec2Embedder())
    for sess in sorted([p for p in in_root.iterdir() if p.is_dir()]):
        n, npy, csv = ing.process_dir(sess, out_root / sess.name)
        print(f"[{sess.name}] {n} files -> {npy if n else '-'}")

if __name__ == "__main__":
    main()
