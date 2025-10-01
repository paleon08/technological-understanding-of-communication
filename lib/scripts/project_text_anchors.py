import os, glob
import numpy as np
import pandas as pd
import umap
from pathlib import Path

IN_DIR = "artifacts/text_anchors"
OUT_SUFFIX = "_2d.csv"

files = sorted(glob.glob(os.path.join(IN_DIR, "*.npy")))
for npy_path in files:
    base = Path(npy_path).stem
    if base.endswith("_2d"):  # 이미 축소된 건 건너뜀
        continue

    print(f"[umap] projecting {base}")
    csv_path = os.path.join(IN_DIR, f"{base}.csv")
    if not os.path.exists(csv_path):
        print(f"  [skip] missing csv: {csv_path}")
        continue

    data = np.load(npy_path)
    df = pd.read_csv(csv_path)

    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(data)

    out = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "text": df["text"],
    })
    out_path = os.path.join(IN_DIR, base + OUT_SUFFIX)
    out.to_csv(out_path, index=False)
    print(f"  -> saved {out_path}")

print("[done] all anchors projected.")
