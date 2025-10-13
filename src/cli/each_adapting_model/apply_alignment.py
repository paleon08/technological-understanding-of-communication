# scripts/align_apply.py
import sys, argparse, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
try:
    from tuc.adapting_model_each_species.alignment import apply_alignment
except ModuleNotFoundError:
    from tuc.adapting_model_each_species.alignment import apply_alignment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="behavior vector .npy")
    ap.add_argument("--W", required=True, help="behavior_to_text_W.npy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    b = np.load(args.input); W = np.load(args.W)
    v = apply_alignment(b, W)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, v.astype("float32"))
    print("ok:", v.shape)

if __name__ == "__main__":
    main()
