# tuc/search.py
from __future__ import annotations
import numpy as np, csv
from .io import load_species_vectors, load_queries, ART
from .encoder import encode_text

def _cosine(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (a.reshape(1, -1) @ B.T).ravel()

def nearest_overall(k: int = 5, queries: list[str] | None = None):
    X, metas = load_species_vectors()
    if X is None:
        raise RuntimeError("No embeddings found. Build or Adjust first.")
    if queries is None:
        queries = load_queries()
    if not queries:
        print("[warn] no queries"); return None

    out = ART / f"nearest_overall_top{k}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query","rank","score","species","name","text","meaning","context"])
        w.writeheader()
        for q in queries:
            qv = encode_text([q])[0]
            sc = _cosine(qv, X); idx = np.argsort(-sc)[:k]
            for rnk, i in enumerate(idx, 1):
                m = metas[i]
                w.writerow({"query": q, "rank": rnk, "score": f"{float(sc[i]):.6f}",
                            "species": m["species"], "name": m["name"], "text": m["text"],
                            "meaning": m["meaning"], "context": m["context"]})
    return out

def nearest_from_vector(vec: np.ndarray, k: int = 5, return_csv: bool = True):
    """전처리된 입력 임베딩(vec)으로 앵커 Top-K 검색."""
    X, metas = load_species_vectors()
    if X is None:
        raise RuntimeError("No embeddings found. Build or Adjust first.")
    # 정규화 가정: vec가 L2-normalized가 아니면 normalize
    v = vec.astype("float32")
    n = np.linalg.norm(v) + 1e-9
    v = v / n
    sc = _cosine(v, X); idx = np.argsort(-sc)[:k]
    rows = []
    for rnk, i in enumerate(idx, 1):
        m = metas[i]
        rows.append({
            "rank": rnk, "score": float(sc[i]),
            "species": m["species"], "name": m["name"], "text": m["text"],
            "meaning": m["meaning"], "context": m["context"],
        })
    if not return_csv:
        return rows

    out = ART / f"nearest_input_top{k}.csv"
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
