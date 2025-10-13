# tuc/adjust.py
from __future__ import annotations
import numpy as np
from ..io import load_anchor_yamls, save_species_vectors, load_adjust
from ..loading_base_model.encoder import encode_text

def _l2n(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)

def _apply_adjust(rows, V, adj):
    if not adj: return V
    scales = np.ones((V.shape[0],), dtype="float32")
    for rule in adj.get("anchor_scales", []):
        rs, rn, s = rule.get("species"), rule.get("name"), float(rule.get("scale",1.0))
        for i, r in enumerate(rows):
            if r["species"]==rs and r["name"]==rn:
                scales[i] *= s
    for rule in adj.get("keyword_boosts", []):
        key, s = str(rule.get("keyword","")).lower(), float(rule.get("scale",1.0))
        if not key: continue
        for i, r in enumerate(rows):
            blob = f'{r.get("text","")} {r.get("meaning","")} {r.get("context","")}'.lower()
            if key in blob: scales[i] *= s
    V2 = (V * scales[:,None]).astype("float32")
    V2 = np.vstack([_l2n(v) for v in V2]).astype("float32")
    return V2

def rebuild_with_adjust():
    """조정 규칙을 적용하여 '다시 임베딩' → 저장.
       (임베딩 자체를 조정하는 정책을 한 곳에서 관리)"""
    adj = load_adjust()
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
        V = encode_text(texts)          # 기본 임베딩
        V = _apply_adjust(rows, V, adj) # 조정만 별도 적용
        save_species_vectors(species, V, rows)
