# scripts/align_fit.py
import sys, argparse, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from tuc.io import ART
try:
    from tuc.alignment import fit_alignment
except ModuleNotFoundError:
    from tuc.alignment import fit_alignment  # 폴백

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="CSV with columns: species,behavior_id,anchor,weight")
    ap.add_argument("--emb-root", required=True, help="root dir of behavior vectors (e.g., artifacts/behaviors/)")
    ap.add_argument("--anchors-csv", default=str(ART/"all_species_text_anchors.csv"))
    ap.add_argument("--out", default=str(ART/"behavior_to_text_W.npy"))
    ap.add_argument("--lambda_", type=float, default=0.1)
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs)

    # anchor 로딩 헬퍼
    import csv
    with open(args.anchors_csv,"r",encoding="utf-8") as f:
        key2idx = {}
        for i, r in enumerate(csv.DictReader(f)):
            key2idx[(r["species"], r["name"])] = i

    import numpy as np
    ALL = np.load(str(ART/"all_species_text_anchors.npy")) if (ART/"all_species_text_anchors.npy").exists() else None
    def load_anchor(species, idx):
        if ALL is None:
            raise SystemExit("all_species_text_anchors.npy not found. Run build first.")
        return ALL[idx]

    Bs, Ts = [], []
    for _, r in pairs.iterrows():
        sp, bid, anch = r["species"], r["behavior_id"], r["anchor"]
        w = float(r.get("weight",1.0))**0.5
        b = np.load(Path(args.emb_root)/sp/f"{bid}.npy")
        t = load_anchor(sp, key2idx[(sp, anch)])
        Bs.append(b*w); Ts.append(t*w)

    B = np.stack(Bs,0); T = np.stack(Ts,0)
    W = fit_alignment(B, T, args.lambda_)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, W); print(f"[OK] saved {args.out}, shape={W.shape}")

if __name__ == "__main__":
    main()
