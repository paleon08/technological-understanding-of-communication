# scripts/proto_make_by_keywords.py
from pathlib import Path
import os, re, numpy as np, pandas as pd

IN_DIR = Path("artifacts/text_anchors")
OUT_DIR = Path("artifacts/prototypes"); OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "vibration": ["vibration","rattle","tail"],
    "defense":   ["defensive","threat","hiss","open-mouth","bluff","flatten"],
    "courtship": ["courtship","mating","bite","quiver"],
    "contact":   ["contact","call","chirp","bark","whistle","squeak","trill","click"],
    "escape":    ["escape","jump","flee","freezing","immobility"],
    "calm":      ["calm","fired-down","exploration","tongue flick"],
}

def make_group_proto(texts, embeds, group, keywords):
    pat = re.compile("|".join(re.escape(k) for k in keywords), re.I)
    mask = np.array([bool(pat.search(t)) for t in texts], dtype=bool)
    if not mask.any(): return None
    X = embeds[mask]
    X = X / (np.linalg.norm(X,axis=1,keepdims=True)+1e-12)
    p = X.mean(0); p /= (np.linalg.norm(p)+1e-12)
    return p, texts[mask]

def main():
    files = sorted([p for p in IN_DIR.glob("*.npy") if not p.name.endswith("_2d.npy")])
    all_texts, all_embeds = [], []
    for npy in files:
        csv = IN_DIR / (npy.stem + ".csv")
        if not csv.exists(): continue
        df = pd.read_csv(csv)
        X = np.load(npy)
        m = min(len(df), len(X))
        all_texts.extend(df["text"].astype(str).tolist()[:m])
        all_embeds.append(X[:m])
    if not all_embeds:
        print("[skip] no anchors"); return
    all_embeds = np.concatenate(all_embeds,0); all_texts = np.array(all_texts, dtype=object)

    summary=[]
    for g, kws in GROUPS.items():
        out = make_group_proto(all_texts, all_embeds, g, kws)
        if out is None: continue
        p, members = out
        np.save(OUT_DIR/f"{g}_proto.npy", p.astype(np.float32))
        pd.Series(members).to_csv(OUT_DIR/f"{g}_members.csv", index=False, header=["text"])
        summary.append({"group": g, "count": len(members)})
    if summary:
        pd.DataFrame(summary).to_csv(OUT_DIR/"prototypes_summary.csv", index=False)
        print("[done] prototypes ->", OUT_DIR)
    else:
        print("[warn] no prototypes created")

if __name__ == "__main__":
    main()
