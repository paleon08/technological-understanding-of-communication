# tuc/rules.py
from __future__ import annotations
import numpy as np, yaml
from pathlib import Path
from ..loading_base_model.encoder import encode_text
from ..io import ART, load_species_matrix_and_meta, save_species_matrix

def _l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    return v / n

def _pull(V: np.ndarray, idx: int, tgt: np.ndarray, step: float):
    V[idx] = _l2n(V[idx] + step * (tgt - V[idx]))

def _push(V: np.ndarray, idx: int, tgt: np.ndarray, step: float):
    V[idx] = _l2n(V[idx] - step * (tgt - V[idx]))

def apply_rules_to_species(rules_path: Path, alpha: float = 0.15, gamma: float = 0.10,
                           out_suffix: str | None = "rules", inplace: bool = False) -> Path:
    r = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8")) or {}
    species = r.get("species")
    if not species:
        raise ValueError(f"{rules_path} lacks 'species'")

    V, meta, name2idx = load_species_matrix_and_meta(species)  # (N,D), meta[dict], {name:idx}

    pos = [p for p in (r.get("positive_pairs") or []) if "text" in p and "anchor" in p]
    neg = [p for p in (r.get("negative_pairs") or []) if "text" in p and "anchor" in p]
    all_texts = [p["text"] for p in pos] + [n["text"] for n in neg]
    E = encode_text(all_texts) if all_texts else np.zeros((0, V.shape[1]), dtype="float32")

    ei = 0
    for p in pos:
        idx = name2idx.get(p["anchor"])
        if idx is None: continue
        w = float(p.get("weight", 1.0))
        _pull(V, idx, E[ei], alpha * w); ei += 1

    for n in neg:
        idx = name2idx.get(n["anchor"])
        if idx is None: continue
        w = float(n.get("weight", 1.0))
        _push(V, idx, E[ei], alpha * w); ei += 1

    def _names2idx(names): return [name2idx[x] for x in names if x in name2idx]

    for grp in (r.get("tie_groups") or []):
        ids = _names2idx(grp)
        if len(ids) >= 2:
            c = _l2n(V[ids].mean(axis=0, keepdims=True))[0]
            for i in ids: V[i] = _l2n(V[i] + gamma * (c - V[i]))

    for grp in (r.get("separate_groups") or []):
        if not (isinstance(grp, list) and len(grp) == 2): continue
        A = _names2idx(grp[0]); B = _names2idx(grp[1])
        if not A or not B: continue
        ca = _l2n(V[A].mean(axis=0, keepdims=True))[0]
        cb = _l2n(V[B].mean(axis=0, keepdims=True))[0]
        for i in A: V[i] = _l2n(V[i] + gamma * (V[i] - cb))
        for i in B: V[i] = _l2n(V[i] + gamma * (V[i] - ca))

    out = save_species_matrix(species, V, suffix=None if inplace else out_suffix)
    return out
