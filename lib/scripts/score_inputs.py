# scripts/score_inputs.py
import argparse, numpy as np, csv, json
from pathlib import Path

def l2n(x, eps=1e-12): n=np.linalg.norm(x,axis=-1,keepdims=True)+eps; return x/n

def load_anchors(npy_path, csv_path):
    A = np.load(npy_path).astype(np.float32)
    meta=[]
    with open(csv_path,'r',encoding='utf-8') as f:
        for row in csv.DictReader(f): meta.append(row)
    assert A.shape[0]==len(meta)
    return l2n(A), meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inputs", default="artifacts/inputs")
    ap.add_argument("--anchors_npy", default="artifacts/text_anchors/anchors.npy")
    ap.add_argument("--anchors_csv", default="artifacts/text_anchors/anchors.csv")
    ap.add_argument("--out", default="artifacts/exp/preds.csv")
    ap.add_argument("-k","--topk", type=int, default=5)
    args=ap.parse_args()

    A, metaA = load_anchors(args.anchors_npy, args.anchors_csv)
    rows=[]
    for npy in Path(args.inputs).glob("*.npy"):
        q = np.load(npy).astype(np.float32)
        q = q[None,:] if q.ndim==1 else q
        q = l2n(q)
        S = (q @ A.T)[0]  # [N]
        idx = np.argsort(-S)[:args.topk]
        jso = json.loads(Path(args.inputs, npy.stem+".json").read_text(encoding="utf-8"))
        for r, i in enumerate(idx,1):
            m = metaA[i]
            rows.append({
                "q_key": npy.stem,
                "rank": r,
                "score": float(S[i]),
                "species": m.get("species",""),
                "name": m.get("name",""),
                "meaning": m.get("meaning",""),
                "tags": m.get("tags",""),
                "q_kind": jso.get("kind",""),
                "q_species": jso.get("species",""),
                "q_src": jso.get("src",""),
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

if __name__=="__main__":
    main()
