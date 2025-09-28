import csv, subprocess, sys, time
from pathlib import Path
import shutil, yaml

cfg = yaml.safe_load(open("configs/download.yaml","r",encoding="utf-8"))
RAW_DIR   = Path(cfg["paths"]["raw_audio"])
CLIPS_DIR = Path(cfg["paths"]["clips_audio"])
META_DIR  = Path(cfg["paths"]["meta"])
SR        = int(cfg["audio"]["target_sr"])
CLIPLEN   = float(cfg["audio"]["clip_len_sec"])
RETRIES   = 3
SLEEP_S   = 1.0
USE_COOKIES = False   # 필요시 True로 바꾸고 아래 브라우저 지정

RAW_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
(META_DIR / "ledger.csv").parent.mkdir(parents=True, exist_ok=True)

manifest = META_DIR / "manifest.tsv"
ledger   = META_DIR / "ledger.csv"
if not manifest.exists():
    raise FileNotFoundError("manifest.tsv 없음. 먼저 매니페스트를 만드세요.")

def which(cmds):
    for c in cmds:
        p = shutil.which(c)
        if p: return p
    return None

def yt_dlp_cmd():
    exe = which(["yt-dlp","yt-dlp.exe"])
    if exe: return [exe]
    return [sys.executable, "-m", "yt_dlp"]

FFMPEG = which(["ffmpeg","ffmpeg.exe"])
if not FFMPEG:
    raise EnvironmentError("ffmpeg 실행파일을 PATH에 추가하세요.")

YTDLP = yt_dlp_cmd()

if not ledger.exists():
    with open(ledger,"w",encoding="utf-8",newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","youtube_id","start","end","label_mapped",
                    "status","local_path","samplerate","duration_sec","notes"])

def run(cmd):
    return subprocess.call(cmd, shell=False)

# --- 교체 시작: 컨테이너 먼저 받고 ffmpeg로 오디오 추출 ---
def download_media(url, tmp_dir, ytid):
    """
    컨테이너 파일(확장자 자유)을 먼저 받는다.
    성공 시 실제 파일 경로(Path)를 리턴.
    """
    FORMAT_TRIES = [
        ["-f", "bestaudio[ext=m4a]/bestaudio/best"],  # m4a가 있으면 가장 성공률 높음
        ["-f", "best"]                                # 통합 스트림도 OK
    ]
    for fmt in FORMAT_TRIES:
        for i in range(RETRIES):
            outtmpl = str(tmp_dir / f"{ytid}.%(ext)s")
            cmd = [*YTDLP, *fmt, "--no-playlist", url, "-o", outtmpl]
            if USE_COOKIES:
                cmd.extend(["--cookies-from-browser","chrome"])
            rc = run(cmd)
            # 다운로드된 실제 파일 찾기(확장자가 무엇이든)
            files = list(tmp_dir.glob(f"{ytid}.*"))
            if rc == 0 and files:
                return True, files[0]
            time.sleep(SLEEP_S)
    return False, None

def extract_and_trim_to_wav(media_path, out_path, start, dur):
    """
    컨테이너(영상/오디오 어떤 형식이든)에서 오디오만 추출해
    10초로 정확 트림(+부족 시 패딩), 48kHz 모노 PCM으로 저장.
    """
    cmd = [FFMPEG, "-y", "-hide_banner",
           "-ss", str(start), "-t", str(dur),
           "-i", str(media_path),
           "-vn",                        # 비디오 무시
           "-ar", str(SR), "-ac", "1",
           "-af", f"apad=pad_dur={dur}", # 짧으면 패딩
           "-c:a", "pcm_s16le",
           str(out_path)]
    rc = run(cmd)
    return rc == 0 and out_path.exists(), f"ffmpeg exit={rc}"
# --- 교체 끝 ---

def trim_to_exact(tmp_path, out_path, start, dur):
    # -t 10으로 정확히, 부족하면 apad로 패딩
    cmd = [FFMPEG,"-y","-hide_banner",
           "-ss", str(start), "-t", str(dur),
           "-i", str(tmp_path),
           "-ar", str(SR), "-ac","1",
           "-af", f"apad=pad_dur={dur}",
           "-c:a","pcm_s16le",
           str(out_path)]
    rc = run(cmd)
    return rc==0 and out_path.exists(), f"ffmpeg exit={rc}"

n_ok = n_fail = 0
with open(manifest,"r",encoding="utf-8") as f, open(ledger,"a",encoding="utf-8",newline="") as g:
    r = csv.DictReader(f, delimiter="\t")
    w = csv.writer(g)
    for row in r:
        ytid  = row["youtube_id"].strip()
        start = float(row["start"])
        end   = float(row["end"])
        label = row["label_mapped"]
        url   = f"https://www.youtube.com/watch?v={ytid}"

        tmp = RAW_DIR / f"{ytid}.wav"
        out = CLIPS_DIR / f"audioset_{ytid}_{int(start)}s_{int(end)}s.wav"
        if out.exists():
            w.writerow(["audioset", ytid, start, start+CLIPLEN, label,
                        "skipped_exists", str(out), SR, CLIPLEN, ""])
            continue

        ok, media = download_media(url, RAW_DIR, ytid)
        status, note = ("downloaded", "") if ok else ("failed_download", "no media")
        if ok:
            ok2, note2 = extract_and_trim_to_wav(media, out, start, CLIPLEN)
            status = "downloaded" if ok2 else "failed_trim"
            note = note2 if not ok2 else ""
        # 정리
        try:
            if ok and media.exists():
                media.unlink()
        except: pass

        dur = CLIPLEN if status=="downloaded" else 0.0
        if status=="downloaded": n_ok += 1
        else: n_fail += 1
        w.writerow(["audioset", ytid, start, start+CLIPLEN, label,
                    status, str(out), SR, dur, note])

print(f"[DONE] ok={n_ok}, fail={n_fail}")
