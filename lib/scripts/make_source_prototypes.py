# scripts/proto_make_source.py
import os, numpy as np
from pathlib import Path

IN_DIR = Path("artifacts/text_anchors")
OUT_DIR = Path("artifacts/source_space"); OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_proto(name: str):
    npy = IN_DIR / f"{name}.npy"
    csv = IN_DIR / f"{name}.csv"
    if not npy.exists() or not csv.exists():
        print(f"[skip] missing: {npy} or {csv}"); return
    X = np.load(npy).astype("float32")
    X = X / (np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
    proto = X.mean(0); proto /= (np.linalg.norm(proto)+1e-12)
    np.save(OUT_DIR/f"{name}_proto.npy", proto)
    print(f"[OK] {name} -> {OUT_DIR}/{name}_proto.npy (D={proto.shape[0]})")

if __name__ == "__main__":
    names = sorted({p.stem for p in IN_DIR.glob("*.npy")} - {p.stem for p in IN_DIR.glob("*_2d.npy")})
    for nm in names:
        make_proto(nm)
