# scripts/metrics.py
import argparse, csv, json
from collections import defaultdict

def load_labels(path):
    lab={}
    with open(path,'r',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            lab[r["q_key"]] = (r["answer_species"], r["answer_name"])
    return lab

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)   # artifacts/exp/preds.csv
    ap.add_argument("--labels", required=True)  # configs/labels.csv
    args=ap.parse_args()

    labels = load_labels(args.labels)
    preds_by_q = defaultdict(list)
    with open(args.preds,'r',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            preds_by_q[r["q_key"]].append(r)

    hit1=hit3=hit5=0; mrr=0; n=0
    for q, plist in preds_by_q.items():
        plist.sort(key=lambda x: int(x["rank"]))
        ans = labels.get(q)
        if not ans: continue
        n+=1
        ranks = [i for i,p in enumerate(plist,1) if (p["species"],p["name"])==ans]
        if ranks:
            r = ranks[0]
            if r==1: hit1+=1
            if r<=3: hit3+=1
            if r<=5: hit5+=1
            mrr += 1.0/r

    out = {
        "n": n,
        "hit@1": hit1/max(n,1),
        "hit@3": hit3/max(n,1),
        "hit@5": hit5/max(n,1),
        "mrr": mrr/max(n,1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
