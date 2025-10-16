# tuc/cli.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .loading_base_model.model_build import build_all      # <- 여기 수정
from .testing.search import nearest_overall, nearest_from_vector  # <- 여기 수정
from .io import ART
import os
if os.getenv("TUC_MODE","text-only")=="text-only":
    pass  # audio 서브커맨드 비활성

def main():
    ap = argparse.ArgumentParser(
        prog="tuc", description="Technological Understanding of Communication CLI"
    )
    ap.add_argument("cmd", choices=["build", "query", "infer"])
    ap.add_argument("--k", type=int, default=5, help="top-k neighbors")
    ap.add_argument("--query", type=str, default=None, help="single query string (overrides configs/queries.txt)")
    ap.add_argument("--input", type=str, default=None, help="vector .npy for infer")
    args = ap.parse_args()

    if args.cmd == "build":
        build_all()
        print("[OK] built anchors")

    if args.cmd == "query":
        if args.query:
            out = nearest_overall(k=args.k, queries=[args.query])
        else:
            out = nearest_overall(k=args.k)
        print(f"[OK] wrote {out}")
        return

    if args.cmd == "infer":
        if not args.input:
            raise SystemExit("--input required (.npy vector)")
        vec = np.load(args.input)
        rows = nearest_from_vector(vec, k=args.k)
        out = ART / f"nearest_input_top{args.k}.csv"
        import pandas as pd
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[OK] wrote {out}")
        return

if __name__ == "__main__":
    main()
