# scripts/make_manifest.py  (robust ver.)
import csv, json, random
from pathlib import Path
import pandas as pd, yaml

cfg = yaml.safe_load(open("configs/download.yaml","r",encoding="utf-8"))
CSV_DIR  = Path(cfg["audioset"]["csv_dir"])
USE_BAL  = bool(cfg["audioset"].get("use_balanced", True))
MAX_SMP  = int(cfg["audioset"].get("max_samples", 20))
META_DIR = Path(cfg["paths"]["meta"])
META_DIR.mkdir(parents=True, exist_ok=True)

# 1) 사용자가 적은 라벨(이름) 목록
user_labels = [s.lower() for s in cfg["labels"]]

# 2) MID<->name 매핑 로드 (class_labels_indices.csv 우선, 없으면 ontology.json)
mid2name = {}
cli = CSV_DIR / "class_labels_indices.csv"
if cli.exists():
    df = pd.read_csv(cli)
    mid2name = dict(zip(df["mid"], df["display_name"]))
else:
    onto = CSV_DIR / "ontology.json"
    if not onto.exists():
        raise FileNotFoundError("ontology.json 또는 class_labels_indices.csv가 필요합니다.")
    j = json.load(open(onto,"r",encoding="utf-8"))
    mid2name = {x["id"]: x["name"] for x in j}

# 3) 이름으로 고른 MID 집합 만들기(느슨한 매칭: 포함관계/소문자 비교)
target_mids = set()
for mid, name in mid2name.items():
    n = name.lower()
    for u in user_labels:
        if (u == n) or (u in n) or (n in u):
            target_mids.add(mid)

# AudioSet의 개 관련 라벨 표기 보완(누락 방지)
# Bark/Howl/Growling/Yip/Bow-wow/Whimper (dog)
for must_name in ["bark","yip","howl","bow-wow","growling","whimper (dog)"]:
    for mid, name in mid2name.items():
        if must_name == name.lower():
            target_mids.add(mid)

if not target_mids:
    raise RuntimeError("선택된 라벨에서 일치하는 MID를 찾지 못했습니다. configs/download.yaml의 labels를 확인하세요.")

# 4) 세그먼트 CSV에서 대상 MID가 하나라도 있으면 채택
seg_csv = CSV_DIR / ("balanced_train_segments.csv" if USE_BAL else "unbalanced_train_segments.csv")

rows=[]
with open(seg_csv,"r",encoding="utf-8") as f:
    r = csv.reader(f)
    for row in r:
        if not row or row[0].startswith("#"):
            continue
        try:
            ytid, start, end, pos = row[0], float(row[1]), float(row[2]), row[3]
        except Exception:
            continue
        # 따옴표 제거하고 MID split
        mids = [m.strip().strip('"') for m in pos.split(",")]
        if set(mids) & target_mids:
            # 사람이 읽는 이름(교집합)도 기록
            names = sorted({mid2name.get(m, m) for m in mids if m in target_mids})
            rows.append((ytid, start, end, "|".join(names)))

random.shuffle(rows)
if MAX_SMP>0:
    rows = rows[:MAX_SMP]

out = META_DIR / "manifest.tsv"
with open(out,"w",encoding="utf-8",newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["youtube_id","start","end","label_mapped"])
    for ytid, s, e, lab in rows:
        w.writerow([ytid, s, e, lab])

print(f"[OK] manifest -> {out} (n={len(rows)})")
print(f" - labels in cfg: {cfg['labels']}")
print(f" - target_mids: {len(target_mids)} mids matched")
