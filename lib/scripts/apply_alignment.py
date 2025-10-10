import argparse, numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)     # 행동 임베딩 벡터 .npy (shape [d_b])
ap.add_argument("--W", required=True)         # behavior_to_text_W.npy
ap.add_argument("--out", default=None)
args = ap.parse_args()

b = np.load(args.input).astype("float32")
if b.ndim>1: b=b.reshape(-1)
W = np.load(args.W).astype("float32")
v = b @ W
v = v / (np.linalg.norm(v)+1e-9)
if args.out:
    np.save(args.out, v.astype("float32"))
print("ok; vec:", v.shape)
