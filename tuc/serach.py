# tuc/search.py
from __future__ import annotations
import numpy as np
from .io import load_species_vectors, load_queries, ART
from .encoder import encode_text
import csv

def nearest_overall(k: int = 5, queries: list[str] | None = None):
    X, metas = load_species_vectors()
    if X is None: 
        raise RuntimeError("No embeddings found. Build or Adjust first.")
    if queries is None:
        queries = load_queries()
    if not queries:
        return
    def cosine(a, B): return (a.reshape(1,-1) @ B.T).ravel()

    out = ART / f"nearest_overall_top{k}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query","rank","score","species","name","text","meaning","context"])
        w.writeheader()
        for q in queries:
            qv = encode_text([q])[0]
            sc = cosine(qv, X); idx = np.argsort(-sc)[:k]
            for rnk, i in enumerate(idx, 1):
                m = metas[i]
                w.writerow({"query":q,"rank":rnk,"score":f"{float(sc[i]):.6f}",
                            "species":m["species"],"name":m["name"],"text":m["text"],
                            "meaning":m["meaning"],"context":m["context"]})
    return out
