# scripts/anchors_project.py
import os, glob
import numpy as np
import pandas as pd
from pathlib import Path

def project_file(npy_path: str, algo: str = "umap", seed: int = 42):
    base = Path(npy_path).stem
    csv_path = os.path.join("artifacts/text_anchors", base + ".csv")
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame({"text":[]})
    X = np.load(npy_path)
    if algo == "svd":
        Xc = X - X.mean(0, keepdims=True)
        import numpy as np
        U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
        reduced = Xc @ Vt[:2].T
    else:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed)
        reduced = reducer.fit_transform(X)
    out = pd.DataFrame({"x": reduced[:,0], "y": reduced[:,1], "text": df.get("text","")})
    out_path = os.path.join("artifacts/text_anchors", base + "_2d.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] {base} -> {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["umap","svd"], default="umap")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for npy in sorted(glob.glob("artifacts/text_anchors/*.npy")):
        if npy.endswith("_2d.npy"): continue
        project_file(npy, algo=args.algo, seed=args.seed)
