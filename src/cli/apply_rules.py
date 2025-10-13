# scripts/apply_rules.py
import sys, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

# 규칙 로직은 패키지에 있음
from tuc.io import load_species_matrix_and_meta, save_species_matrix
try:
    from tuc.rules import apply_rules_to_species   # 패키지 경로
except ModuleNotFoundError:
    from tuc.rules import apply_rules_to_species   # 동일(레거시 호환)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, help="e.g., crestedgecko or cornsnake")
    ap.add_argument("--rules", required=True, help="rules file or folder (e.g., configs/rules/crestedgecko_rules.yml)")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=0.05)
    ap.add_argument("--out-suffix", default="rules")  # 저장 파일명에 붙일 태그
    args = ap.parse_args()

    X, meta, name2idx = load_species_matrix_and_meta(args.species)
    X2 = apply_rules_to_species(X, meta, args.rules, alpha=args.alpha, gamma=args.gamma)
    save_species_matrix(args.species, X2, meta, suffix=args.out_suffix)
    print(f"[OK] rules applied -> artifacts/text_anchors/{args.species}_text_anchors.{args.out_suffix}.npy/.csv")

if __name__ == "__main__":
    main()
