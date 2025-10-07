from pathlib import Path
import os, re, numpy as np, pandas as pd

IN_DIR = Path("artifacts/text_anchors")
OUT_DIR = Path("artifacts/prototypes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 키워드 -> 그룹 이름 매핑 (원하면 추가/수정)
GROUPS = {
    "vibration": ["vibration", "rattle", "tail"],
    "defense":   ["defensive", "threat", "hiss", "open-mouth", "bluff", "flatten"],
    "courtship": ["courtship", "mating", "bite", "quiver"],
    "contact":   ["contact", "call", "chirp", "bark", "whistle", "squeak", "trill", "click"],
    "escape":    ["escape", "jump", "flee", "freezing", "immobility"],
    "calm":      ["calm", "fired-down", "exploration", "tongue flick"],
}

def l2norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / n

def make_group_proto(texts, embeds, group_name, keywords):
    mask = np.array([
        any(re.search(kw, t, flags=re.I) for kw in keywords)
        for t in texts
    ], dtype=bool)
    if mask.sum() == 0:
        return None
    Z = l2norm(embeds[mask])
    p = l2norm(Z.mean(axis=0, keepdims=True)).squeeze(0)
    return p, texts[mask]

def main():
    # 모든 앵커 파일 합치기 (종/파일 구분 없이)
    all_texts, all_embeds = [], []
    for csv_path in IN_DIR.glob("*.csv"):
        npy_path = csv_path.with_suffix(".npy")
        if not npy_path.exists(): 
            continue
        df = pd.read_csv(csv_path)
        X = np.load(npy_path)
        # 길이 맞추기 안전장치
        m = min(len(df), len(X))
        all_texts.extend(df["text"].astype(str).tolist()[:m])
        all_embeds.append(X[:m])

    if not all_texts:
        print("[skip] no anchor embeddings found in artifacts/text_anchors")
        return

    all_embeds = np.concatenate(all_embeds, axis=0)
    all_texts = np.array(all_texts, dtype=object)

    # 그룹별 프로토타입 생성
    summary = []
    for g, kws in GROUPS.items():
        out = make_group_proto(all_texts, all_embeds, g, kws)
        if out is None:
            print(f"[skip] {g}: no matches")
            continue
        p, matched_texts = out
        np.save(OUT_DIR / f"{g}_proto.npy", p.astype(np.float32))
        pd.Series(matched_texts).to_csv(OUT_DIR / f"{g}_members.csv", index=False, header=["text"])
        summary.append({"group": g, "count": len(matched_texts)})

    if summary:
        pd.DataFrame(summary).to_csv(OUT_DIR / "prototypes_summary.csv", index=False)
        print("[done] prototypes saved to:", OUT_DIR)
    else:
        print("[warn] no prototypes created (no keyword matches)")

if __name__ == "__main__":
    main()
