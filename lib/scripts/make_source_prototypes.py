import os, numpy as np, pandas as pd
from pathlib import Path

IN_DIR  = "artifacts/text_anchors"
OUT_DIR = "artifacts/source_space"
os.makedirs(OUT_DIR, exist_ok=True)

def make_proto(name: str):
    npy = Path(IN_DIR) / f"{name}.npy"
    csv = Path(IN_DIR) / f"{name}.csv"
    if not npy.exists() or not csv.exists():
        print(f"[skip] missing: {npy} or {csv}"); return
    X = np.load(npy)                  # [N, D]
    # L2 정규화 평균(0-벡터 방지)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    proto = X.mean(axis=0)
    proto /= (np.linalg.norm(proto) + 1e-12)
    np.save(Path(OUT_DIR)/f"{name}_proto.npy", proto.astype(np.float32))
    print(f"[done] {name} -> {OUT_DIR}/{name}_proto.npy (D={proto.shape[0]})")

if __name__ == "__main__":
    for sp in ["crestedgecko","cornsnake"]:
        make_proto(sp)
