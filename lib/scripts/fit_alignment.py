import argparse, numpy as np, pandas as pd
from pathlib import Path

def l2n(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True)+1e-9)

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)  # CSV
ap.add_argument("--emb-root", required=True)  # artifacts/behaviors/{species}/
ap.add_argument("--anchors-csv", default="artifacts/text_anchors/all_species_text_anchors.csv")
ap.add_argument("--out", default="artifacts/text_anchors/behavior_to_text_W.npy")
ap.add_argument("--lambda_", type=float, default=0.1)
args = ap.parse_args()

pairs = pd.read_csv(args.pairs)
A = pd.read_csv(args.anchors_csv)  # columns: species,name,text,meaning,context,index, ... + maybe x,y
# 각 species+name -> 앵커 인덱스
key2idx = {(r["species"], r["name"]): int(r["index"]) for _, r in A.iterrows()}
# 앵커 임베딩 행렬 스택 (.npy per species)
# 간단화: species별 npy를 모아 하나로 스택한 파일이 이미 있을 경우 사용
# 여기서는 species별 로드 예시
def load_anchor_matrix():
    # 전종 스택 파일이 있다면 그걸 로드하는 경로로 바꿔도 됨
    # 예: artifacts/text_anchors/all_species_text_anchors.npy
    # 지금은 species별 로드 예시
    by_species = {}
    for sp in set(pairs["species"].tolist()):
        p = Path(f"artifacts/text_anchors/{sp}_text_anchors.npy")
        by_species[sp] = np.load(p)
    # 인덱스가 species별이므로, 전종 일괄 비교 시에는 메타와 함께 따로 관리 필요
    return by_species

by_sp = load_anchor_matrix()

Bs, Ts = [], []
for _, r in pairs.iterrows():
    sp = r["species"]; bid = r["behavior_id"]; anch = r["anchor"]; w = float(r.get("weight",1.0))
    b = np.load(Path(args.emb_root) / sp / f"{bid}.npy").astype("float32")
    b = l2n(b.reshape(1,-1))[0]
    idx = key2idx[(sp, anch)]
    t = l2n(by_sp[sp][idx:idx+1])[0]
    # 가중치는 나중에 곱
    Bs.append(b*np.sqrt(w))
    Ts.append(t*np.sqrt(w))

B = np.stack(Bs, axis=0); T = np.stack(Ts, axis=0)
lb = args.lambda_
# ridge 해(안정형)
BTB = B.T @ B; BTt = B.T @ T
W = np.linalg.solve(BTB + lb*np.eye(B.shape[1], dtype=B.dtype), BTt)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
np.save(args.out, W.astype("float32"))
print(f"[OK] W saved -> {args.out}, shape={W.shape}")
