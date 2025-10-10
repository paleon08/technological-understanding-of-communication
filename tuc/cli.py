# tuc/cli.py
from __future__ import annotations
import argparse
from .model_build import build_all
from .adjust import rebuild_with_adjust
from .search import nearest_overall, nearest_from_vector
import numpy as np, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build","adjust","query","infer","run"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--input", type=str, default=None)  # vector .npy
    args = ap.parse_args()

    if args.cmd == "infer":
        if not args.input: raise SystemExit("--input required (.npy vector)")
        vec = np.load(args.input)
        rows = nearest_from_vector(vec, k=args.k)
        from pathlib import Path
        out = (Path("artifacts/text_anchors") / f"nearest_input_top{args.k}.csv")
        import pandas as pd
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[OK] wrote {out}")
        return

    # ... 기존 build/adjust/query/run 유지 ...


    if args.cmd == "build":
        build_all()

    elif args.cmd == "adjust":
        rebuild_with_adjust()

    elif args.cmd == "query":
        qs = [args.query] if args.query else None
        out = nearest_overall(k=args.k, queries=qs)
        if out: print(f"[OK] wrote {out}")

    elif args.cmd == "run":
        build_all()              # A
        rebuild_with_adjust()    # B
        out = nearest_overall(k=args.k)  # use configs/queries.txt if present
        if out: print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()
