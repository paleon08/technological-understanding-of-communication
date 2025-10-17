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
    ei += len(pos)
    # DEPRECATED: geometric push disabled (anchors are fixed)
    ei += len(neg)

    def _names2idx(names): return [name2idx[x] for x in names if x in name2idx]

    
    out = save_species_matrix(species, V, meta, suffix=None if inplace else out_suffix)
    return out
