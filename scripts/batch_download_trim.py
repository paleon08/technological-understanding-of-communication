# scripts/batch_download_trim.py
import csv, subprocess, time
from pathlib import Path
import yaml

cfg = yaml.safe_load(open("configs/download.yaml","r",encoding="utf-8"))
RAW_DIR   = Path(cfg["paths"]["raw_audio"])
CLIPS_DIR = Path(cfg["paths"]["clips_audio"])
META_DIR  = Path(cfg["paths"]["meta"])
SR        = int(cfg["audio"]["target_sr"])
CLIPLEN   = float(cfg["audio"]["clip_len_sec"])
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
(META_DIR / "ledger.csv").parent.mkdir(parents=True, exist_ok=True)

manifest = META_DIR / "manifest.tsv"
ledger   = META_DIR / "ledger.csv"
if not manifest.exists():
    raise FileNotFoundError("manifest.tsv 없음. 먼저 make_manifest.py를 실행하세요.")

# ledger 헤더 보장
if not ledger.exists():
    with open(ledger,"w",encoding="utf-8",newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","youtube_id","start","end","label_mapped",
                    "status","local_path","samplerate","duration_sec","notes"])

def run(cmd):
    return subprocess.call(cmd, shell=False)

n_ok, n_fail = 0, 0
with open(manifest,"r",encoding="utf-8") as f, open(ledger,"a",encoding="utf-8",newline="") as g:
    reader = csv.DictReader(f, delimiter="\t")
    w = csv.writer(g)
    for row in reader:
        ytid = row["youtube_id"].strip()
        start= float(row["start"]); end=float(row["end"])
        label= row["label_mapped"]
        url  = f"https://www.youtube.com/watch?v={ytid}"

        tmp_path = RAW_DIR / f"{ytid}.wav"  # 임시 원본
        out_path = CLIPS_DIR / f"audioset_{ytid}_{int(start)}s_{int(end)}s.wav"

        status, note = "downloaded", ""

        if out_path.exists():  # 이미 처리된 건 스킵
            w.writerow(["audioset", ytid, start, start+CLIPLEN, label,
                        "skipped_exists", str(out_path), SR, CLIPLEN, ""])
            continue

        # 1) yt-dlp로 오디오 추출(wav)
        cmd1 = ["yt-dlp","-f","bestaudio","--extract-audio",
                "--audio-format","wav","--audio-quality","0",
                url,"-o",str(tmp_path)]
        if run(cmd1)!=0 or not tmp_path.exists():
            status, note = "failed_download", "yt-dlp error"
        else:
            # 2) ffmpeg로 정확 구간 트림 + 48kHz/mono
            cmd2 = ["ffmpeg","-y","-hide_banner",
                    "-ss",str(start),"-to",str(start+CLIPLEN),
                    "-i",str(tmp_path),
                    "-ar",str(SR),"-ac","1","-c:a","pcm_s16le",
                    str(out_path)]
            if run(cmd2)!=0 or not out_path.exists():
                status, note = "failed_trim", "ffmpeg error"

        # 임시 원본 정리(용량 아끼기)
        try:
            if tmp_path.exists(): tmp_path.unlink()
        except: pass

        if status=="downloaded":
            n_ok+=1; dur=CLIPLEN
        else:
            n_fail+=1; dur=0.0

        w.writerow(["audioset", ytid, start, start+CLIPLEN, label,
                    status, str(out_path), SR, dur, note])

print(f"[DONE] ok={n_ok}, fail={n_fail}")
