# scripts/eval_score_inputs.py
import argparse, numpy as np, csv, json
from pathlib import Path

def l2n(x, eps=1e-12): n=np.linalg.norm(x,axis=-1,keepdims=True)+eps; return x/n

def load_anchors(npy_path, csv_path):
    A = np.load(npy_path).astype(np.float32); A = l2n(A)
    meta=[]
    with open(csv_path,'r',encoding='utf-8') as f:
        for row in csv.DictReader(f): meta.append(row)
    assert A.shape[0]==len(meta), "npy and csv length mismatch"
    return A, meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--anchors-npy", required=True)
    ap.add_argument("--anchors-csv", required=True)
    ap.add_argument("--inputs-dir", required=True, help="dir with *.npy and optional *.json meta")
    ap.add_argument("--out", required=True, help="CSV path for ranked results")
    ap.add_argument("--k", type=int, default=5)
    args=ap.parse_args()

    A, meta = load_anchors(args.anchors_npy, args.anchors_csv)
    rows=[]
    for npy in sorted(Path(args.inputs_dir).glob("*.npy")):
        v = l2n(np.load(npy).reshape(1,-1))[0]
        S = (v.reshape(1,-1) @ A.T).ravel()
        idx = np.argsort(-S)[:args.k]
        jmeta = Path(npy.with_suffix(".json"))
        jso = json.loads(jmeta.read_text(encoding="utf-8")) if jmeta.exists() else {}
        for r, i in enumerate(idx, start=1):
            m = meta[i]
            rows.append({
                "q_key": npy.stem, "rank": r, "score": float(S[i]),
                "species": m.get("species",""), "name": m.get("name",""),
                "meaning": m.get("meaning",""), "context": m.get("context",""),
                "q_kind": jso.get("kind",""), "q_species": jso.get("species",""), "q_src": jso.get("src",""),
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(args.out,"w",encoding="utf-8",newline="") as f:
        w=_csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[OK] wrote {args.out}, rows={len(rows)}")

if __name__=="__main__":
    main()
