# tuc/cli.py
from __future__ import annotations
import argparse, os, csv, json, re
from pathlib import Path
import numpy as np

from .loading_base_model.model_build import build_all
from .testing.search import nearest_overall, nearest_from_vector
from .io import ART

# ---- 유틸 ----
def _canon_species(s: str) -> str:
    s0 = (s or "").strip().lower()
    m = {
        "cre": "crestedgecko",
        "crestedgecko": "crestedgecko",
        "crested": "crestedgecko",
        "gecko": "crestedgecko",
        "con": "cornsnake",
        "cornsnake": "cornsnake",
        "corn": "cornsnake",
        "snake": "cornsnake",
    }
    return m.get(s0, s0)

def _read_csv(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except UnicodeError:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

def _trunc(s: str | None, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")

def _get(d: dict, *keys, default=""):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _print_rows(rows: list[dict]):
    if not rows:
        print("[info] 결과가 없습니다.")
        return
    header = f"{'#':>2} {'name':<22} {'score':>6}  {'meaning'}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        name    = _get(r, "name", "anchor")
        score   = _get(r, "score")
        try: score_f = float(score)
        except Exception: score_f = float("nan")
        meaning = _trunc(_get(r, "meaning", "text", "desc", "description"), 80)
        print(f"{i:>2} {name:<22.22} {score_f:6.3f}  {meaning}")

# ---- 메인 ----
def main():
    ap = argparse.ArgumentParser(
        prog="tuc", description="Technological Understanding of Communication CLI"
    )
    ap.add_argument("cmd", choices=["build", "query", "infer"])

    # 최소 옵션만 유지
    ap.add_argument("--species", type=str, help="필수. 예: CRE / CON 또는 crestedgecko / cornsnake")
    ap.add_argument("--behavior", type=str, help="필수. 행동(자연어) 쿼리")
    ap.add_argument("--k", type=int, default=5, help="top-k (기본 5)")
    ap.add_argument("--input", type=str, default=None, help="infer 용: .npy 벡터 경로")

    args = ap.parse_args()

    if args.cmd == "build":
        build_all()
        print("[OK] built anchors")
        return

    if args.cmd == "query":
        if not args.species or not args.behavior:
            raise SystemExit("사용법: tuc query --species CRE --behavior \"tail vibration\" [--k 5]")

        species = _canon_species(args.species)
        behavior = args.behavior

        # 1) 전체 스택에서 넉넉히 뽑아온 뒤
        k_internal = max(args.k * 10, 50)  # 필터 후에도 k를 채우도록 여유
        out_csv = nearest_overall(k=k_internal, queries=[behavior])

        # 2) 방금 생성된 CSV에서 해당 종만 필터
        rows_all = _read_csv(Path(out_csv))
        rows_sp = [r for r in rows_all
                   if r.get("query") == behavior and _canon_species(r.get("species","")) == species]

        # 3) 상위 k만 출력
        rows_top = rows_sp[: args.k]
        _print_rows(rows_top)

        # 파일 경로도 참고용으로 알려줌
        print(f"[OK] wrote {out_csv}")
        return

    if args.cmd == "infer":
        if not args.input:
            raise SystemExit("--input required (.npy vector)")
        vec = np.load(args.input)
        rows = nearest_from_vector(vec, k=args.k)
        out = ART / f"nearest_input_top{args.k}.csv"

        # 저장
        import pandas as pd
        pd.DataFrame(rows).to_csv(out, index=False)

        # 화면 출력(간단 표)
        _print_rows(rows[: args.k])
        print(f"[OK] wrote {out}")
        return

if __name__ == "__main__":
    # audio 서브커맨드는 현재 비활성
    if os.getenv("TUC_MODE", "text-only") == "text-only":
        pass
    main()
