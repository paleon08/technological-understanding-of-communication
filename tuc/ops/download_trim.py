import subprocess
from pathlib import Path
from lib.io_utils import append_ledger

def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd, shell=False)

def download_and_trim_row(row, raw_dir: Path, out_dir: Path, ledger_path: Path,
                          sr: int = 48000, clip_len: float = 10.0):
    ytid = str(row["youtube_id"])
    start = float(row["start"]); end = float(row["end"])
    label = str(row.get("label_mapped","Dog"))
    dur = max(0.1, min(clip_len, end - start))

    out_dir.mkdir(parents=True, exist_ok=True); raw_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / f"audioset_{ytid}_{int(start)}s_{int(start+dur)}s.wav"
    if out_wav.exists():
        append_ledger({"source":"audioset","youtube_id":ytid,"start":start,"end":start+dur,
                       "label":label,"status":"skip","out_path":str(out_wav),
                       "sr":sr,"cliplen":dur,"note":"exists"}, ledger_path)
        return

    # 1) download
    tmp = raw_dir / f"{ytid}.%(ext)s"
    url = f"https://www.youtube.com/watch?v={ytid}"
    rc = _run(["yt-dlp","-f","ba","-o", str(tmp), "--no-part","--no-progress", url])
    if rc != 0:
        append_ledger({"source":"audioset","youtube_id":ytid,"start":start,"end":start+dur,
                       "label":label,"status":"fail","out_path":"", "sr":sr,"cliplen":dur,
                       "note":f"yt-dlp rc={rc}"}, ledger_path)
        return

    got = None
    for ext in ("m4a","webm","mp4","opus","mp3"):
        cand = raw_dir / f"{ytid}.{ext}"
        if cand.exists(): got = cand; break
    if got is None:
        hits = list(raw_dir.glob(f"{ytid}.*"))
        got = hits[0] if hits else None
    if got is None:
        append_ledger({"source":"audioset","youtube_id":ytid,"start":start,"end":start+dur,
                       "label":label,"status":"fail","out_path":"", "sr":sr,"cliplen":dur,
                       "note":"downloaded file not found"}, ledger_path)
        return

    # 2) trim
    rc = _run([
        "ffmpeg","-hide_banner","-loglevel","error",
        "-ss", str(start), "-t", str(dur), "-i", str(got),
        "-ac","1","-ar", str(sr), "-vn","-sn", str(out_wav)
    ])
    try: got.unlink()
    except Exception: pass

    if rc == 0 and out_wav.exists():
        append_ledger({"source":"audioset","youtube_id":ytid,"start":start,"end":start+dur,
                       "label":label,"status":"ok","out_path":str(out_wav),
                       "sr":sr,"cliplen":dur,"note":""}, ledger_path)
    else:
        append_ledger({"source":"audioset","youtube_id":ytid,"start":start,"end":start+dur,
                       "label":label,"status":"fail","out_path":str(out_wav),
                       "sr":sr,"cliplen":dur,"note":f"ffmpeg rc={rc}"}, ledger_path)
