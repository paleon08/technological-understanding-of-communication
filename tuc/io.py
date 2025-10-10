# tuc/io.py
from __future__ import annotations
from pathlib import Path
import yaml, csv, numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG  = ROOT / "configs"
ART  = ROOT / "artifacts" / "text_anchors"

def load_anchor_yamls():
    """Yield (species, anchors:list[dict|str]) from configs/anchors/*.yml"""
    for p in sorted((CFG / "anchors").glob("*.yml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        species = data.get("species") or p.stem
        anchors = data.get("anchors") or []
        yield species, anchors

def save_species_vectors(species: str, V: np.ndarray, rows: list[dict]):
    """Save per-species anchor embeddings and CSV metadata."""
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / f"{species}_text_anchors.npy", V)
    for i, r in enumerate(rows):
        r.setdefault("index", i)
        r.setdefault("species", species)
    with (ART / f"{species}_text_anchors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["species","name","text","meaning","context","index"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ["species","name","text","meaning","context","index"]})

def write_global_2d(metas: list[dict], Z: np.ndarray):
    """Write 2D projection CSV for quick visualization across species."""
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / "all_species_text_anchors_2d.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["species","name","text","meaning","context","x","y"])
        w.writeheader()
        for meta, z in zip(metas, Z):
            row = {
                "species": meta.get("species",""),
                "name": meta.get("name",""),
                "text": meta.get("text",""),
                "meaning": meta.get("meaning",""),
                "context": meta.get("context",""),
                "x": f"{float(z[0]):.6f}",
                "y": f"{float(z[1]):.6f}",
            }
            w.writerow(row)

def load_adjust() -> dict:
    p = CFG / "adjust.yml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def load_queries() -> list[str]:
    p = CFG / "queries.txt"
    if not p.exists():
        return []
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]

def load_species_matrix_and_meta(species: str):
    Xp = ART / f"{species}_text_anchors.npy"
    Cp = ART / f"{species}_text_anchors.csv"
    if not Xp.exists() or not Cp.exists():
        raise FileNotFoundError(f"Build missing for species={species}: {Xp.name}, {Cp.name}")
    X = np.load(Xp).astype("float32")
    meta = []
    with Cp.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            meta.append(row)
    name2idx = {m["name"]: int(m.get("index", i)) for i, m in enumerate(meta)}
    return X, meta, name2idx

def save_species_matrix(species: str, X: np.ndarray, meta: list[dict], suffix: str | None = None):
    """Save an updated species matrix and metadata; optionally with a suffix marker."""
    ART.mkdir(parents=True, exist_ok=True)
    tag = "" if not suffix else f".{suffix}"
    np.save(ART / f"{species}_text_anchors{tag}.npy", X.astype("float32"))
    with (ART / f"{species}_text_anchors{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["species","name","text","meaning","context","index"])
        w.writeheader()
        for i, m in enumerate(meta):
            m2 = dict(m); m2["index"] = i
            w.writerow(m2)

def load_species_vectors():
    """Load all species embeddings stacked -> (X_all, meta_all) or (None, None) if not built."""
    items = sorted(ART.glob("*_text_anchors.npy"))
    if not items:
        return None, None
    Xs, metas = [], []
    for npy in items:
        species = npy.name.split("_text_anchors.npy")[0]
        X = np.load(npy).astype("float32")
        metas_csv = ART / f"{species}_text_anchors.csv"
        cur = []
        if metas_csv.exists():
            import csv
            with metas_csv.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    cur.append(row)
        else:
            for i in range(len(X)):
                cur.append({"species": species, "name": f"anchor_{i:03d}", "text":"", "meaning":"", "context":"", "index": i})
        Xs.append(X); metas.extend(cur)
    X_all = np.vstack(Xs).astype("float32")
    return X_all, metas

# 공개 상수: 아티팩트 루트 경로
__all__ = ["ROOT", "CFG", "ART", "load_anchor_yamls", "save_species_vectors", "write_global_2d",
           "load_adjust", "load_queries", "load_species_matrix_and_meta", "save_species_matrix",
           "load_species_vectors"]
