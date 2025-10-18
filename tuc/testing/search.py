# tuc/testing/search.py
from __future__ import annotations
import os, csv
from pathlib import Path
import numpy as np
import yaml

from ..io import load_species_matrix_and_meta, load_species_vectors, load_queries, ART, CFG
from ..loading_base_model.encoder import encode_text

# ====== Tunables (env 로도 설정 가능) ======
RULE_PRIOR_WEIGHT = float(os.environ.get("TUC_RULE_PRIOR_WEIGHT", "0.35"))
ADJUST_PRIOR_WEIGHT = float(os.environ.get("TUC_ADJUST_PRIOR_WEIGHT", "0.15"))

# ====== Basic cosine (dot if L2-normalized) ======
def _cosine(q: np.ndarray, X: np.ndarray) -> np.ndarray:
    q = q.reshape(1, -1).astype("float32")
    return (q @ X.T).ravel()

# ====== Load prior tables from YAMLs ======
def _normalize_meaning_name(name: str) -> str:
    name = (name or "").strip().strip('"').strip("'")
    return name if name.endswith("_meaning") else f"{name}_meaning"

def _load_rules_prior_table() -> dict[str, float]:
    """
    Aggregate positive_pairs weights across all rule files into {anchor_name(*_meaning): weight_sum}.
    """
    pri: dict[str, float] = {}
    rules_dir = CFG / "rules"
    if not rules_dir.exists():
        return pri

    for yml in rules_dir.glob("*_rules.yml"):
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        pairs = data.get("positive_pairs") or []
        if not isinstance(pairs, list):
            continue
        for row in pairs:
            if not isinstance(row, dict):
                continue
            a = _normalize_meaning_name(str(row.get("anchor", "")))
            w = float(row.get("weight", 1.0))
            pri[a] = pri.get(a, 0.0) + w
    return pri

def _load_adjust_prior_table(alias_to_meaning: dict[str, str] | None = None) -> dict[str, float]:
    """
    Interpret adjust.yml anchor_scales as {anchor_name(*_meaning): weight_sum}.
    If adjust uses legacy (alias) names, normalize to *_meaning via alias map.
    """
    pri: dict[str, float] = {}
    yml = CFG / "adjust.yml"
    if not yml.exists():
        return pri
    try:
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
    except Exception:
        return pri

    rows = data.get("anchor_scales") or data.get("weights") or data.get("adjust") or []
    if not isinstance(rows, list):
        return pri
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("name") or row.get("anchor") or row.get("id") or "")
        # alias -> meaning 정규화
        if alias_to_meaning and raw in alias_to_meaning:
            a = alias_to_meaning[raw]
        else:
            a = _normalize_meaning_name(raw)
        try:
            w = float(row.get("scale") or row.get("weight") or 0.0)
        except Exception:
            w = 0.0
        pri[a] = pri.get(a, 0.0) + w
    return pri

# ====== Build per-anchor prior vector aligned to metas ======
def _build_prior_vector(metas: list[dict],
                        rules_prior: dict[str, float],
                        adjust_prior: dict[str, float]) -> np.ndarray:
    pri = np.zeros((len(metas),), dtype=np.float32)
    for i, m in enumerate(metas):
        name = str(m.get("name", ""))
        pri[i] = rules_prior.get(name, 0.0) * RULE_PRIOR_WEIGHT \
               + adjust_prior.get(name, 0.0) * ADJUST_PRIOR_WEIGHT
    return pri

# ====== Optional alias map (if metas include aliases in metadata) ======
def _build_alias_map_from_configs() -> dict[str, str]:
    """
    Build alias->meaning map by scanning anchors/*.yml.
    alias가 존재하면 adjust 등의 레거시 이름을 의미 앵커명으로 바꿀 때 사용.
    """
    alias2meaning: dict[str, str] = {}
    anchors_dir = CFG / "anchors"
    if not anchors_dir.exists():
        return alias2meaning
    for yml in anchors_dir.glob("*.yml"):
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        for it in (data.get("anchors") or []):
            if not isinstance(it, dict):
                continue
            mname = _normalize_meaning_name(str(it.get("name", "")))
            for al in it.get("aliases", []) or []:
                alias2meaning[str(al)] = mname
            # legacy 호환: name에 _meaning이 빠져 있던 경우도 alias로 취급
            base = str(it.get("name","")).replace("_meaning","")
            if base and base != mname:
                alias2meaning[base] = mname
    return alias2meaning

# ====== Public APIs ======
def nearest_overall(k: int = 5, queries: list[str] | None = None):
    """
    Export top-k nearest anchors for each query from configs/queries.txt to CSV.
    Output: artifacts/text_anchors/nearest_overall_top{k}.csv
    """
    X, metas = load_species_vectors()
    if X is None:
        raise RuntimeError("No embeddings found. Build species anchors first.")
    if queries is None:
        queries = load_queries()
    if not queries:
        raise RuntimeError("No queries provided (configs/queries.txt missing or empty).")

    # prior 준비
    alias_map = _build_alias_map_from_configs()
    rules_prior = _load_rules_prior_table()
    adjust_prior = _load_adjust_prior_table(alias_map)

    out_rows = []
    for qtxt in queries:
        qv = encode_text(qtxt).astype("float32")
        sim = _cosine(qv, X)
        pri = _build_prior_vector(metas, rules_prior, adjust_prior)
        sc = sim + pri

        idx = np.argsort(-sc)[:k]
        for rnk, i in enumerate(idx, start=1):
            m = metas[i]
            out_rows.append({
                "query": qtxt,
                "rank": rnk,
                "score": float(sc[i]),
                "species": m.get("species",""),
                "name": m.get("name",""),        # *_meaning
                "text": m.get("text",""),
                "meaning": m.get("meaning",""),
                "context": m.get("context",""),
            })

    ART.mkdir(parents=True, exist_ok=True)
    out_path = ART / f"nearest_overall_top{k}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query","rank","score","species","name","text","meaning","context"])
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    return str(out_path)

def nearest_from_vector(vec: np.ndarray, k: int = 5):
    """
    Return list[dict] of top-k for a single vector input (already normalized preferred).
    """
    X, metas = load_species_vectors()
    if X is None:
        raise RuntimeError("No embeddings found. Build species anchors first.")
    v = vec.reshape(-1).astype("float32")

    alias_map = _build_alias_map_from_configs()
    rules_prior = _load_rules_prior_table()
    adjust_prior = _load_adjust_prior_table(alias_map)

    sim = _cosine(v, X)
    pri = _build_prior_vector(metas, rules_prior, adjust_prior)
    sc = sim + pri

    idx = np.argsort(-sc)[:k]
    out = []
    for rnk, i in enumerate(idx, start=1):
        m = metas[i]
        out.append({
            "rank": rnk,
            "score": float(sc[i]),
            "species": m.get("species",""),
            "name": m.get("name",""),
            "text": m.get("text",""),
            "meaning": m.get("meaning",""),
            "context": m.get("context",""),
        })
    return out
