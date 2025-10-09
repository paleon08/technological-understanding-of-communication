# tuc/model_build.py
from __future__ import annotations
import numpy as np
from .encoder import encode_text
from .io import load_anchor_yamls, save_species_vectors, write_global_2d

def _svd2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def build_all():
    metas_all, mats = [], []
    for species, anchors in load_anchor_yamls():
        texts, rows = [], []
        for i, a in enumerate(anchors):
            if isinstance(a, str):
                rows.append({"name": f"anchor_{i:03d}", "text": a, "meaning": "", "context": "", "species": species})
                texts.append(a)
            else:
                t = a.get("text") or a.get("phrase") or a.get("anchor") or ""
                if not t: continue
                rows.append({"name": a.get("name", f"anchor_{i:03d}"),
                             "text": t, "meaning": a.get("meaning",""),
                             "context": a.get("context",""), "species": species})
                texts.append(t)
        if not texts: 
            continue
        V = encode_text(texts)
        save_species_vectors(species, V, rows)
        mats.append(V); metas_all += [{**r, "index": i} for i, r in enumerate(rows)]
    if not mats:
        return [], None
    X = np.vstack(mats)
    Z = _svd2d(X)
    write_global_2d(metas_all, Z)
    return metas_all, X
