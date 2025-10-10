import sys, argparse, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tuc.alignment import apply_alignment

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)  # behavior vector .npy
ap.add_argument("--W", required=True)      # behavior_to_text_W.npy
ap.add_argument("--out", default=None)
args = ap.parse_args()

b = np.load(args.input); W = np.load(args.W)
v = apply_alignment(b, W)
if args.out:
    np.save(args.out, v.astype("float32"))
print("ok:", v.shape)
