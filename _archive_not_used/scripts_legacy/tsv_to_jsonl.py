import csv, json
from pathlib import Path
import yaml

cfg = yaml.safe_load(open("configs/download.yaml","r",encoding="utf-8"))
META = Path(cfg["paths"]["meta"])
CLIPS= Path(cfg["paths"]["clips_audio"])

spldir = META / "splits"
split_files = {k: (spldir/f"{k}.txt").read_text(encoding="utf-8").splitlines()
               for k in ["train","val","test"] if (spldir/f"{k}.txt").exists()}
yt2split = {yt:"train" for yt in split_files.get("train",[])}
for yt in split_files.get("val",[]): yt2split[yt]="val"
for yt in split_files.get("test",[]): yt2split[yt]="test"

out = META / "manifest.jsonl"
with open(META/"manifest.tsv","r",encoding="utf-8") as fin, \
     open(out,"w",encoding="utf-8") as fout:
    r = csv.DictReader(fin, delimiter="\t")
    for row in r:
        ytid  = row["youtube_id"].strip()
        start = float(row["start"]); end = float(row["end"])
        labels= [s.strip() for s in row["label_mapped"].split("|")]
        audio = CLIPS / f"audioset_{ytid}_{int(start)}s_{int(end)}s.wav"
        ex = {
            "id": f"audioset_{ytid}_{int(start)}_{int(end)}",
            "ytid": ytid,
            "start": start, "end": end,
            "labels": labels,
            "audio_path": str(audio),
            "video_path": None,
            "frames_glob": None,
            "captions": [],
            "split": yt2split.get(ytid, "train")
        }
        json.dump(ex, fout, ensure_ascii=False)
        fout.write("\n")
print(f"[OK] {out}")
