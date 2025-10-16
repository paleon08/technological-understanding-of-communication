# src/cli/each_adapting_model/apply_rules.py
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from tuc.adapting_model_each_species.rules import apply_rules_to_species

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True, help="configs/rules/<species>_rules.yml")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--out-suffix", default="rules")
    args = ap.parse_args()
    out = apply_rules_to_species(args.rules, alpha=args.alpha, gamma=args.gamma, out_suffix=args.out_suffix)
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()
