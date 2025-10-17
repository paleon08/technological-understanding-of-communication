# tuc/model_build.py
from __future__ import annotations
import numpy as np
from .encoder import encode_text
from ..io import load_anchor_yamls, save_species_vectors, write_global_2d

def _svd2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return (Xc @ Vt[:2].T).astype("float32")

def build_all():
    """
    Build embeddings for all species anchors from configs/anchors/*.yml
    - 각 종별로 NPY/CSV 저장
    - 전체 스택 2D 투영 CSV도 생성
    """
    metas_all, mats = [], []
    for species, anchors in load_anchor_yamls():
        texts, rows = [], []
        def _normalize_meaning_name(name: str) -> str:
            name = (name or "").strip()
            return name if name.endswith("_meaning") else f"{name}_meaning"

        for i, a in enumerate(anchors):
            if isinstance(a, str):
                # 문자열로 된 앵커는 의미 텍스트로 간주
                nm = _normalize_meaning_name(f"anchor_{i:03d}")
                rows.append({"name": nm, "text": a, "meaning": a, "context": "", "species": species})
                texts.append(a)
            else:
                # 의미 전용 스키마 우선: canonical_text -> (fallback) meaning -> (legacy) text/phrase/anchor
                t = a.get("canonical_text") or a.get("meaning") or a.get("text") or a.get("phrase") or a.get("anchor") or ""
                if not t:
                    continue
                raw_nm = a.get("name", f"anchor_{i:03d}")
                nm = _normalize_meaning_name(raw_nm)
                rows.append({
                    "name": nm,
                    "text": t,
                    "meaning": t,                 # 의미 텍스트로 채움
                    "context": a.get("context",""),
                    "species": species,
                })
                texts.append(t)
        if not texts:
            continue
        V = encode_text(texts)
        save_species_vectors(species, V, rows)
        metas_all.extend(rows)
        mats.append(V)
    if not mats:
        return
    X_all = np.vstack(mats).astype("float32")
    Z2 = _svd2d(X_all)
    write_global_2d(metas_all, Z2)
