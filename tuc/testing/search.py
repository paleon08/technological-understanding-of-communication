# tuc/search.py
from __future__ import annotations
import numpy as np, csv
from ..io import load_species_vectors, load_queries, ART
from ..loading_base_model.encoder import encode_text

def _cosine(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (a.reshape(1, -1) @ B.T).ravel()

def nearest_overall(k: int = 5, queries: list[str] | None = None):
    """
    Export top-k nearest anchors for each query in configs/queries.txt -> CSV.
    Output: artifacts/text_anchors/nearest_overall_top{k}.csv
    """
    X, metas = load_species_vectors()
    if X is None:
        raise RuntimeError("No embeddings found. Build or Adjust first.")
    if queries is None:
        queries = load_queries()
    if not queries:
        raise RuntimeError("No queries to run. Add lines to configs/queries.txt")

    out = ART / f"nearest_overall_top{k}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query","rank","score","species","name","text","meaning","context"])
        w.writeheader()
        for q in queries:
            qv = encode_text([q])[0]
            sc = _cosine(qv, X); idx = np.argsort(-sc)[:k]
            for rnk, i in enumerate(idx, start=1):
                m = metas[i]
                w.writerow({
                    "query": q,
                    "rank": rnk,
                    "score": f"{float(sc[i]):.6f}",
                    "species": m.get("species",""),
                    "name": m.get("name",""),
                    "text": m.get("text",""),
                    "meaning": m.get("meaning",""),
                    "context": m.get("context",""),
                })
    return out

def nearest_from_vector(vec: np.ndarray, k: int = 5):
    """
    Return list[dict] for a single vector input (already normalized preferred).
    """
    X, metas = load_species_vectors()
    if X is None:
        raise RuntimeError("No embeddings found. Build or Adjust first.")
    v = vec.reshape(-1).astype("float32")
    n = np.linalg.norm(v) + 1e-9
    v = v / n
    sc = (v.reshape(1,-1) @ X.T).ravel()
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
