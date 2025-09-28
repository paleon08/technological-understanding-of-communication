# src/tuc/cli.py
import sys, subprocess
from pathlib import Path
import typer

app = typer.Typer(help="TUC thin CLI (wraps existing scripts/*)")

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

def run(pyfile, *args):
    cmd = [sys.executable, str(SCRIPTS / pyfile), *args]
    print("[RUN]", " ".join(cmd))
    rc = subprocess.call(cmd, shell=False)
    if rc != 0:
        raise SystemExit(rc)

@app.command(help="Build manifest (wraps make_manifest_balanced.py)")
def manifest(config: str = "configs/download.yaml"):
    # 스크립트가 --config를 지원하지 않으면 인자 없이 재시도
    try:
        run("make_manifest_balanced.py", "--config", config)
    except SystemExit:
        run("make_manifest_balanced.py")

@app.command(help="Download & trim (wraps batch_download_trim.py)")
def download():
    run("batch_download_trim.py")

@app.command(help="Make splits (wraps make_splits.py)")
def splits():
    run("make_splits.py")

@app.command(help="Convert TSV->JSONL (wraps tsv_to_jsonl.py)")
def jsonl():
    run("tsv_to_jsonl.py")

@app.command(help="Train baseline (wraps train_audio_baseline.py)")
def train(epochs: int = 1, bs: int = 8):
    run("train_audio_baseline.py", "--epochs", str(epochs), "--bs", str(bs))

if __name__ == "__main__":
    app()
