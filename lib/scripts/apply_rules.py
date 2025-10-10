# 규칙(자연어) → CLAP 텍스트 임베딩 → 앵커 벡터를 미세 이동(끌어당김/밀어냄)
import argparse, yaml, numpy as np
from pathlib import Path

# CLAP 텍스트 임베더 (이미 교체한 파일을 그대로 재사용)
from .tuc.encoder import encode_text
from .tuc.io import ART

def l2n(v): 
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    return v / n

def load_species_artifacts(species):
    V = np.load(ART / f"{species}_text_anchors.npy")          # (N, D)
    meta = []
    import csv
    with open(ART / f"{species}_text_anchors.csv", "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr: meta.append(r)
    # name -> index
    name2idx = {r["name"]: int(r["index"]) for r in meta}
    return V, meta, name2idx

def apply_positive(V, idx, target_vec, step):
    V[idx] = l2n(V[idx] + step * (target_vec - V[idx]))

def apply_negative(V, idx, target_vec, step):
    V[idx] = l2n(V[idx] - step * (target_vec - V[idx]))

def group_pull(V, indices, gamma):
    if len(indices) < 2: return
    centroid = l2n(V[indices].mean(axis=0, keepdims=True))[0]
    for i in indices:
        V[i] = l2n(V[i] + gamma * (centroid - V[i]))

def group_push(V, A, B, gamma):
    if not A or not B: return
    ca = l2n(V[A].mean(axis=0, keepdims=True))[0]
    cb = l2n(V[B].mean(axis=0, keepdims=True))[0]
    # 서로 반대 방향으로 소폭 이동
    for i in A:
        V[i] = l2n(V[i] + gamma * (V[i] - cb))
    for i in B:
        V[i] = l2n(V[i] + gamma * (V[i] - ca))

def run_for_species(rules_path, alpha, gamma, out_suffix, inplace):
    rules = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8")) or {}
    species = rules.get("species")
    assert species, f"rules file {rules_path} must specify species"
    V, meta, name2idx = load_species_artifacts(species)

    # 1) positive/negative pairs
    pos = rules.get("positive_pairs", [])
    neg = rules.get("negative_pairs", [])
    # 해당 텍스트 문구들을 한 번에 임베딩
    pos_texts = [p["text"] for p in pos if "text" in p and "anchor" in p]
    neg_texts = [n["text"] for n in neg if "text" in n and "anchor" in n]
    all_texts = pos_texts + neg_texts
    if all_texts:
        E = encode_text(all_texts)  # (M, D) L2 정규화됨
    else:
        E = np.zeros((0, V.shape[1]), dtype="float32")

    # 인덱스 매핑
    ei = 0
    for p in pos:
        if "text" not in p or "anchor" not in p: continue
        anchor = p["anchor"]; w = float(p.get("weight", 1.0))
        if anchor not in name2idx: continue
        idx = name2idx[anchor]
        apply_positive(V, idx, E[ei], alpha * w); ei += 1

    for n in neg:
        if "text" not in n or "anchor" not in n: continue
        anchor = n["anchor"]; w = float(n.get("weight", 1.0))
        if anchor not in name2idx: continue
        idx = name2idx[anchor]
        apply_negative(V, idx, E[ei], alpha * w); ei += 1

    # 2) 그룹 규칙
    def names_to_indices(names):
        return [name2idx[n] for n in names if n in name2idx]

    for grp in (rules.get("tie_groups") or []):
        group_pull(V, names_to_indices(grp), gamma)

    for grp in (rules.get("separate_groups") or []):
        if isinstance(grp, list) and len(grp) == 2 and isinstance(grp[0], list) and isinstance(grp[1], list):
            group_push(V, names_to_indices(grp[0]), names_to_indices(grp[1]), gamma)

    # 저장
    out_name = f"{species}_text_anchors{'' if inplace else f'.{out_suffix}'}"
    np.save(ART / f"{out_name}.npy", V)
    print(f"[OK] {species}: saved -> {ART / (out_name + '.npy')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", type=str, required=True, help="rules folder or single yaml")
    ap.add_argument("--alpha", type=float, default=0.15, help="pair step size")
    ap.add_argument("--gamma", type=float, default=0.10, help="group step size")
    ap.add_argument("--out-suffix", type=str, default="rules", help="suffix for output npy")
    ap.add_argument("--inplace", action="store_true", help="overwrite original npy")
    args = ap.parse_args()

    p = Path(args.rules)
    paths = [p] if p.is_file() else sorted(p.glob("*_rules.yml"))
    if not paths:
        raise SystemExit(f"No rules found under {p}")
    for y in paths:
        run_for_species(y, args.alpha, args.gamma, args.out_suffix, args.inplace)

if __name__ == "__main__":
    main()
