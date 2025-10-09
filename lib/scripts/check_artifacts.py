# scripts/check_artifacts.py
from pathlib import Path
import numpy as np, pandas as pd

SPECIES = ["crestedgecko", "cornsnake"]
TA_DIR = Path("artifacts/text_anchors")
SS_DIR = Path("artifacts/source_space")

def check_text_anchors(name):
    npy = TA_DIR / f"{name}.npy"
    csv = TA_DIR / f"{name}.csv"
    if not npy.exists() or not csv.exists():
        return f"[MISS] {name}: {npy.name} or {csv.name} missing"
    X = np.load(npy)          # [N, D]
    df = pd.read_csv(csv)     # columns: text, ...
    msg = []
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] <= 10:
        msg.append(f"[BAD] {name}: shape={X.shape}")
    if len(df) != X.shape[0]:
        msg.append(f"[BAD] {name}: rows(csv)={len(df)} != N(npy)={X.shape[0]}")
    if np.isnan(X).any():
        msg.append(f"[BAD] {name}: NaN found")
    # 중복 앵커 텍스트 체크
    if "text" in df.columns:
        dups = df["text"].duplicated().sum()
        if dups > 0:
            msg.append(f"[WARN] {name}: duplicated anchor names={dups}")
    if not msg:
        msg.append(f"[OK] {name}: N={X.shape[0]}, D={X.shape[1]}")
    return " | ".join(msg)

def check_proto(name, expect_dim=None):
    npy = SS_DIR / f"{name}_proto.npy"
    if not npy.exists():
        return f"[MISS] {name}: proto missing"
    p = np.load(npy)
    if p.ndim != 1:
        return f"[BAD] {name}: proto ndim={p.ndim}, shape={p.shape}"
    if expect_dim and p.shape[0] != expect_dim:
        return f"[BAD] {name}: proto D={p.shape[0]} != anchors D={expect_dim}"
    # 거의 단위벡터?
    n = float(np.linalg.norm(p))
    if not (0.9 <= n <= 1.1):
        return f"[WARN] {name}: proto norm={n:.3f} (expected ~1)"
    return f"[OK] {name}: proto D={p.shape[0]}"

if __name__ == "__main__":
    TA_DIR.mkdir(parents=True, exist_ok=True)
    SS_DIR.mkdir(parents=True, exist_ok=True)
    print("== TEXT ANCHORS ==")
    dims = {}
    for sp in SPECIES:
        m = check_text_anchors(sp)
        print(m)
        if "[OK]" in m:
            X = np.load(TA_DIR / f"{sp}.npy")
            dims[sp] = X.shape[1]
    print("\n== PROTOTYPES ==")
    for sp in SPECIES:
        d = dims.get(sp)
        print(check_proto(sp, expect_dim=d))
