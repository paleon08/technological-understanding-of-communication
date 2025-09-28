# 실행: python scripts\make_manifest_dog.py
import sys, pathlib
# lib 경로 추가
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "lib"))

from pathlib import Path
import pandas as pd
from lib.io_utils import write_manifest, MANIFEST_COLUMNS
from lib.ingest.audioset_dog import build_manifest

AUDIOSET_DIR = Path("data/meta/audioset")  # class_labels_indices.csv 등이 있는 폴더
OUT_PATH     = Path("data/meta/manifest.tsv")
PER_LABEL    = 200  # 라벨당 최대 샘플 수

def main():
    df = build_manifest(str(AUDIOSET_DIR), per_label_quota=PER_LABEL, seed=42)
    df = df[MANIFEST_COLUMNS]
    write_manifest(df, OUT_PATH)
    print(f"[OK] manifest -> {OUT_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    main()
