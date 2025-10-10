# scripts/batch_ingest.py
import argparse, os, glob, json, numpy as np
from pathlib import Path
from tuc.ingest import Ingestor
from tuc.encoder import Projector

EXT_AUDIO = {".wav",".flac",".m4a",".aac"}
EXT_VIDEO = {".mp4",".mov",".mkv"}
EXT_NUMPY = {".npy",".npz",".csv"}

def save_q(q, meta, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f"{meta['species']}_{meta['event']}_{abs(hash(meta['src']))%10**8:08d}"
    np.save(out_dir/f"{key}.npy", q.astype(np.float32))
    with open(out_dir/f"{key}.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return key

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="project/data")
    ap.add_argument("--out", default="artifacts/inputs")
    args=ap.parse_args()

    ing, pj = Ingestor(), Projector()
    out_dir = Path(args.out)

    for species in ("cre","con"):
        for session in sorted(Path(args.root, species).glob("*")):
            # notes.jsonl가 있으면 free_text도 q로 저장
            notes = Path(session,"notes.jsonl")
            if notes.exists():
                for line in notes.read_text(encoding="utf-8").splitlines():
                    if not line.strip(): continue
                    rec = json.loads(line)
                    payload = ing.from_text(rec.get("free_text","").strip() or rec["event"])
                    q = pj.text(payload)
                    meta = {"kind":"text","species":species,"event":rec.get("event",""),"src":str(notes)}
                    save_q(q, meta, out_dir)

            # 파일 스캔
            for p in glob.glob(str(Path(session,"**","*")), recursive=True):
                pth = Path(p)
                if not pth.is_file(): continue
                ext = pth.suffix.lower()
                try:
                    if ext in EXT_AUDIO:
                        payload = ing.from_audio_file(str(pth))
                        q = pj.audio(payload); kind="audio"
                    elif ext in EXT_VIDEO:
                        payload = ing.from_video_file(str(pth))
                        q = pj.video(payload); kind="video"
                    elif ext in EXT_NUMPY:
                        # 센서 파일은 csv/npy로 처리(있을 때)
                        if ext == ".npy":
                            import numpy as np
                            arr = np.load(pth).astype(np.float32)
                        elif ext == ".csv":
                            import csv, numpy as np
                            rows=[]
                            with open(pth,'r',encoding='utf-8') as f:
                                for r in csv.reader(f):
                                    if r: rows.append([float(x) for x in r])
                            arr = np.asarray(rows,dtype=np.float32)
                        else:
                            continue
                        payload = ing.from_sensor_array(arr)
                        q = pj.sensor(payload); kind="sensor"
                    else:
                        continue
                    meta={"kind":kind,"species":species,"event":"unknown","src":str(pth)}
                    save_q(q, meta, out_dir)
                except Exception as e:
                    print("[skip]", pth, "->", e)

if __name__=="__main__":
    main()
