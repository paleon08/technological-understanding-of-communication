# lib/scripts/apply_rules.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from tuc.rules import apply_rules_to_species

ap = argparse.ArgumentParser()
ap.add_argument("--rules", required=True, help="rules file or folder (…/_rules.yml)")
ap.add_argument("--alpha", type=float, default=0.15)
ap.add_argument("--gamma", type=float, default=0.10)
ap.add_argument("--out-suffix", type=str, default="rules")
ap.add_argument("--inplace", action="store_true")
args = ap.parse_args()

p = Path(args.rules)
paths = [p] if p.is_file() else sorted(p.glob("*_rules.yml"))
if not paths:
    raise SystemExit(f"No rules found under {p}")
for y in paths:
    out = apply_rules_to_species(y, alpha=args.alpha, gamma=args.gamma,
                                 out_suffix=args.out_suffix, inplace=args.inplace)
    print(f"[OK] {y.name} -> {out}")
