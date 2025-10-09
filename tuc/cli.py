# tuc/cli.py
from __future__ import annotations
import argparse
from .model_build import build_all
from .adjust import rebuild_with_adjust
from .search import nearest_overall

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build","adjust","query","run"], help="""
    build = 모델 생성(앵커→임베딩)
    adjust = 종별 조정(규칙 적용 재빌드)
    query = queries.txt 또는 --query로 검색
    run   = build → adjust → query 순서로 원샷
    """)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--query", type=str, default="")
    args = ap.parse_args()

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
