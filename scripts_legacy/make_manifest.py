import csv, json, random
from pathlib import Path
import pandas as pd, yaml
from collections import defaultdict

cfg = yaml.safe_load(open("configs/download.yaml","r",encoding="utf-8"))
CSV_DIR  = Path(cfg["audioset"]["csv_dir"])
META_DIR = Path(cfg["paths"]["meta"])
META_DIR.mkdir(parents=True, exist_ok=True)

USER_LABELS = [s.lower() for s in cfg["labels"]]
QUOTA = int(cfg["audioset"].get("per_label_quota", 100))
SEED  = int(cfg["audioset"].get("seed", 42))
random.seed(SEED)

# --- MID<->이름 매핑 로드 ---
mid2name = {}
cli = CSV_DIR / "class_labels_indices.csv"
if cli.exists():
    df = pd.read_csv(cli)
    mid2name = dict(zip(df["mid"], df["display_name"]))
else:
    onto = CSV_DIR / "ontology.json"
    if not onto.exists():
        raise FileNotFoundError("ontology.json 또는 class_labels_indices.csv 필요")
    j = json.load(open(onto,"r",encoding="utf-8"))
    mid2name = {x["id"]: x["name"] for x in j}

# 라벨 이름으로 MID 집합 만들기(느슨한 매칭 + 개 라벨 보완)
target_mids = set()
for mid, name in mid2name.items():
    n = name.lower()
    if any((u==n) or (u in n) or (n in u) for u in USER_LABELS):
        target_mids.add(mid)
for must in ["bark","yip","howl","bow-wow","growling","whimper (dog)"]:
    for mid, name in mid2name.items():
        if name.lower()==must: target_mids.add(mid)

def load_segments(csv_path):
    rows=[]
    with open(csv_path,"r",encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row or row[0].startswith("#"): continue
            try:
                ytid, start, end, pos = row[0], float(row[1]), float(row[2]), row[3]
            except Exception:
                continue
            mids = [m.strip().strip('"') for m in pos.split(",")]
            inter = set(mids) & target_mids
            if inter:
                hit_names = sorted({mid2name.get(m,m) for m in inter})
                rows.append((ytid, start, end, hit_names, mids))
    random.shuffle(rows)
    return rows

balanced_csv   = CSV_DIR / "balanced_train_segments.csv"
unbalanced_csv = CSV_DIR / "unbalanced_train_segments.csv"
bal = load_segments(balanced_csv)
unb = load_segments(unbalanced_csv)

# 라벨별 후보 목록 구성(행은 중복될 수 있음)
label2cands = defaultdict(list)
for src_name_list, pool in [(["balanced"], bal), (["unbalanced"], unb)]:
    for ytid, s, e, names, mids in pool:
        for nm in names:
            label2cands[nm].append((ytid, s, e, "|".join(names)))

# 균형 추출(중복 행 방지)
selected = {}
picked_keys = set()
for label in cfg["labels"]:
    want = QUOTA
    cands = label2cands.get(label, [])
    random.shuffle(cands)
    cnt = 0
    for ytid, s, e, lab in cands:
        key = (ytid, s, e)
        if key in picked_keys: 
            continue
        picked_keys.add(key)
        selected[key] = (ytid, s, e, lab)
        cnt += 1
        if cnt >= want: break

# 결과 저장
rows = list(selected.values())
random.shuffle(rows)
out = META_DIR / "manifest.tsv"
with open(out,"w",encoding="utf-8",newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["youtube_id","start","end","label_mapped"])
    for ytid, s, e, lab in rows:
        w.writerow([ytid, s, e, lab])

print(f"[OK] manifest -> {out} (total {len(rows)} clips)")
for label in cfg["labels"]:
    print(f" - {label:<10} : {sum(label in r[3].split('|') for r in rows)}")
