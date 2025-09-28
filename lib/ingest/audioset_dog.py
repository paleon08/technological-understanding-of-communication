from pathlib import Path
import pandas as pd
import random

DOG_LABELS = ["Bark","Dog","Yip","Howl","Whimper","Growling","Pant","Bow-wow"]

def _read_segments_csv(csv_path: Path) -> pd.DataFrame:
    # AudioSet CSV는 앞에 '#' 주석이 있음
    return pd.read_csv(csv_path, comment="#", header=None,
                       names=["YTID","start","end","labels"]).dropna()

def _load_label_map(class_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(class_csv)
    return df[["index","mid","display_name"]]

def build_manifest(audioset_dir: str, per_label_quota: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    audioset_dir 안에는 아래 파일들이 있다고 가정:
      - class_labels_indices.csv
      - balanced_train_segments.csv / unbalanced_train_segments.csv / eval_segments.csv
    """
    audioset_dir = Path(audioset_dir)
    label_map = _load_label_map(audioset_dir / "class_labels_indices.csv")
    # 개 관련 표시 이름 → mid 목록
    mids = label_map[label_map["display_name"].isin(DOG_LABELS)]["mid"].tolist()
    if not mids:
        raise RuntimeError("개 관련 라벨을 class_labels_indices.csv에서 찾지 못했습니다.")

    seg_files = [
        audioset_dir / "balanced_train_segments.csv",
        audioset_dir / "unbalanced_train_segments.csv",
        audioset_dir / "eval_segments.csv",
    ]
    dfs = []
    for f in seg_files:
        if f.exists():
            dfs.append(_read_segments_csv(f))
    if not dfs:
        raise RuntimeError("AudioSet 세그먼트 CSV를 찾지 못했습니다.")
    seg = pd.concat(dfs, ignore_index=True)

    # labels 컬럼은 "/m/..." 가 콤마로 묶여 있음 → 개 관련 mid 포함 행만 필터
    mask = seg["labels"].astype(str).apply(lambda s: any(mid in s for mid in mids))
    seg = seg[mask].copy()

    # label_mapped: 가장 먼저 매칭되는 display_name 하나로 단순화
    mid2name = {row["mid"]: row["display_name"] for _, row in label_map.iterrows()}
    def map_first_name(s: str):
        for mid in mids:
            if mid in s:
                return mid2name[mid]
        return "Dog"
    seg["label_mapped"] = seg["labels"].apply(map_first_name)

    # 라벨별 quota 샘플링
    random.seed(seed)
    outs = []
    for name in seg["label_mapped"].unique():
        df = seg[seg["label_mapped"]==name]
        if len(df) > per_label_quota:
            outs.append(df.sample(per_label_quota, random_state=seed))
        else:
            outs.append(df)
    seg2 = pd.concat(outs, ignore_index=True)

    # 최종 매니페스트 컬럼 정리
    out = seg2[["YTID","start","end","label_mapped"]].rename(
        columns={"YTID":"youtube_id"})
    out = out.drop_duplicates(subset=["youtube_id","start","end"])
    return out
