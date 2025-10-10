# tuc/cli.py
from __future__ import annotations
import argparse
from .model_build import build_all
from .adjust import rebuild_with_adjust
from .search import nearest_overall
from .search import nearest_overall, nearest_from_vector
import numpy as np, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build","adjust","query","infer","run"], help="...")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--input", type=str, default=None)  # infer용
    args = ap.parse_args()

    if args.cmd == "infer":
        if not args.input:
            raise SystemExit("--input 벡터(.npy) 경로가 필요합니다.")
        vec = np.load(args.input)
        if vec.ndim > 1:
            vec = vec.reshape(-1)  # [D] 로 맞추기
        out = nearest_from_vector(vec, k=args.k)
        if out: print(f"[OK] wrote {out}")
        return

    # 기존 build/adjust/query/run 그대로...


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
