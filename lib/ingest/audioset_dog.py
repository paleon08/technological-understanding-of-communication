from pathlib import Path
import pandas as pd, random

DOG_LABELS = ["Bark","Dog","Yip","Howl","Whimper","Growling","Pant","Bow-wow"]

def _read_segments_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, comment="#", header=None,
                       names=["YTID","start","end","labels"]).dropna()

def _load_label_map(class_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(class_csv)
    return df[["index","mid","display_name"]]

def build_manifest(audioset_dir: str, per_label_quota: int = 200, seed: int = 42) -> pd.DataFrame:
    d = Path(audioset_dir)
    label_map = _load_label_map(d / "class_labels_indices.csv")
    mids = label_map[label_map["display_name"].isin(DOG_LABELS)]["mid"].tolist()
    if not mids:
        raise RuntimeError("class_labels_indices.csv에서 개 관련 라벨을 못 찾았습니다.")

    seg_files = [d/"balanced_train_segments.csv", d/"unbalanced_train_segments.csv", d/"eval_segments.csv"]
    dfs = [ _read_segments_csv(f) for f in seg_files if f.exists() ]
    if not dfs: raise RuntimeError("AudioSet 세그먼트 CSV를 찾지 못했습니다.")
    seg = pd.concat(dfs, ignore_index=True)

    mask = seg["labels"].astype(str).apply(lambda s: any(mid in s for mid in mids))
    seg = seg[mask].copy()

    mid2name = {r["mid"]: r["display_name"] for _, r in label_map.iterrows()}
    def map_first_name(s: str):
        for mid in mids:
            if mid in s: return mid2name[mid]
        return "Dog"
    seg["label_mapped"] = seg["labels"].apply(map_first_name)

    random.seed(seed)
    outs = []
    for name in seg["label_mapped"].unique():
        df = seg[seg["label_mapped"] == name]
        outs.append(df.sample(min(len(df), per_label_quota), random_state=seed))
    seg2 = pd.concat(outs, ignore_index=True)

    out = seg2[["YTID","start","end","label_mapped"]].rename(columns={"YTID":"youtube_id"})
    return out.drop_duplicates(subset=["youtube_id","start","end"])
