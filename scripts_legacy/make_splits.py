import csv, random
from pathlib import Path
import yaml

cfg = yaml.safe_load(open("configs/download.yaml","r",encoding="utf-8"))
META_DIR = Path(cfg["paths"]["meta"])
SEED = int(cfg["audioset"].get("seed", 42))
random.seed(SEED)

manifest = META_DIR / "manifest.tsv"
ytids = []
with open(manifest,"r",encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\t")
    for row in r:
        ytids.append(row["youtube_id"].strip())

ytids = sorted(set(ytids))
random.shuffle(ytids)

n = len(ytids)
n_train = int(n*0.8)
n_val   = int(n*0.1)
train = ytids[:n_train]
val   = ytids[n_train:n_train+n_val]
test  = ytids[n_train+n_val:]

spldir = META_DIR / "splits"
spldir.mkdir(parents=True, exist_ok=True)
for name, ids in [("train",train),("val",val),("test",test)]:
    (spldir / f"{name}.txt").write_text("\n".join(ids), encoding="utf-8")

print(f"[OK] splits -> {spldir} (train/val/test = {len(train)}/{len(val)}/{len(test)})")
