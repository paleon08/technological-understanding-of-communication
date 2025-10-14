# tuc/cli.py — minimal wrapper CLI
from __future__ import annotations
import argparse, subprocess, sys, shlex

def _run(cmd: str) -> int:
    print(f"[tuc] $ {cmd}")
    return subprocess.call(cmd, shell=True)

def _do_build() -> int:
    # 레포 기준 기본 빌드 스크립트 경로 시도(필요에 맞게 바꿔도 됨)
    candidates = [
        "python src\\cli\\model_loading\\embed_text_anchors.py",
        "python scripts\\embed_text_anchors.py",
    ]
    for c in candidates:
        if _run(c) == 0: return 0
    print("[tuc] build script not found. Adjust path in tuc/cli.py.")
    return 1

def _do_query(qs: list[str], k: int) -> int:
    if not qs: 
        print("[tuc] --query 가 필요합니다."); 
        return 2
    quoted = " ".join([f"--query {shlex.quote(q)}" for q in qs])
    candidates = [
        f"python src\\cli\\testing\\query_infer.py {quoted} --k {k}",
        f"python scripts\\query_infer.py {quoted} --k {k}",
    ]
    for c in candidates:
        if _run(c) == 0: return 0
    print("[tuc] query script not found. Adjust path in tuc/cli.py.")
    return 1

def _do_infer(inp: str, k: int) -> int:
    candidates = [
        f"python src\\cli\\testing\\query_infer.py --input {shlex.quote(inp)} --k {k}",
        f"python scripts\\query_infer.py --input {shlex.quote(inp)} --k {k}",
    ]
    for c in candidates:
        if _run(c) == 0: return 0
    print("[tuc] infer script not found. Adjust path in tuc/cli.py.")
    return 1

def main():
    ap = argparse.ArgumentParser(prog="tuc", description="Technological Understanding of Communication CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Embed anchors → artifacts/text_anchors/")
    b.set_defaults(fn=lambda a: _do_build())

    q = sub.add_parser("query", help="Nearest search for text queries against anchors")
    q.add_argument("--query", "-q", action="append", help="text query (repeatable)")
    q.add_argument("--k", type=int, default=5)
    q.set_defaults(fn=lambda a: _do_query(a.query, a.k))

    inf = sub.add_parser("infer", help="Nearest search from a saved vector (.npy)")
    inf.add_argument("--input", required=True)
    inf.add_argument("--k", type=int, default=5)
    inf.set_defaults(fn=lambda a: _do_infer(a.input, a.k))

    args = ap.parse_args()
    sys.exit(args.fn(args))

if __name__ == "__main__":
    main()
