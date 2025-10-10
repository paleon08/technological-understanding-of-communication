import sys, argparse, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tuc.alignment import fit_alignment
from tuc.io import ART

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)       # CSV: species,behavior_id,anchor,weight
ap.add_argument("--emb-root", required=True)    # artifacts/behaviors/
ap.add_argument("--anchors-csv", default=str(ART/"all_species_text_anchors.csv"))
ap.add_argument("--out", default=str(ART/"behavior_to_text_W.npy"))
ap.add_argument("--lambda_", type=float, default=0.1)
args = ap.parse_args()

pairs = pd.read_csv(args.pairs)
A = pd.read_csv(args.anchors_csv)  # columns: species,name,index,...
key2idx = {(r["species"], r["name"]): int(r["index"]) for _, r in A.iterrows()}

def load_anchor(sp, idx):
    M = np.load(ART / f"{sp}_text_anchors.npy")
    return M[idx]

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
