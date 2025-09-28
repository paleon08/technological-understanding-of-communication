import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from pathlib import Path
from lib.io_utils import read_manifest
from lib.ops.download_trim import download_and_trim_row

MANIFEST = Path("data/meta/manifest.tsv")
RAW_DIR  = Path("data/raw"); CLIPS_DIR = Path("data/clips")
LEDGER   = Path("data/meta/ledger.csv")

def main():
    df = read_manifest(MANIFEST)
    print(f"[INFO] rows: {len(df)}")
    for _, row in df.iterrows():
        download_and_trim_row(row, RAW_DIR, CLIPS_DIR, LEDGER, sr=48000, clip_len=10.0)

if __name__ == "__main__":
    main()
