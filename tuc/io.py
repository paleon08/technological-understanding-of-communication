# tuc/io.py
from __future__ import annotations
from pathlib import Path
import yaml, csv, numpy as np

ROOT = Path(__file__).resolve().parents[1]         # 프로젝트 루트(OK)
CFG  = ROOT / "configs"                            # ❌ ROOT.parent 아님
ART  = ROOT / "artifacts" / "text_anchors"         # ❌ ROOT.parent 아님

def load_anchor_yamls():
    for p in sorted((CFG / "anchors").glob("*.yml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        species = data.get("species") or p.stem
        anchors = data.get("anchors") or []
        yield species, anchors

def save_species_vectors(species: str, V: np.ndarray, rows: list[dict]):
    ART.mkdir(parents=True, exist_ok=True)
    np.save(ART / f"{species}_text_anchors.npy", V)
    with (ART / f"{species}_text_anchors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["species","name","text","meaning","context","index"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "species": species, "index": i})

def load_species_vectors():
    """모든 종 임베딩과 메타를 로드."""
    mats, metas = [], []
    for csv_path in sorted(ART.glob("*_text_anchors.csv")):
        species = csv_path.stem.replace("_text_anchors", "")
        npy_path = ART / f"{species}_text_anchors.npy"
        if not npy_path.exists():
            continue
        V = np.load(npy_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
        if len(rows) != V.shape[0]:
            print(f"[WARN] meta/vec mismatch: {species}")
        mats.append(V); metas.extend(rows)
    if not mats:
        return None, []
    return np.vstack(mats).astype("float32"), metas

def write_global_2d(metas: list[dict], Z: np.ndarray):
    ART.mkdir(parents=True, exist_ok=True)
    with (ART / "all_species_text_anchors_2d.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["species","name","text","meaning","context","index","x","y"])
        w.writeheader()
        for meta, z in zip(metas, Z):
            w.writerow({**meta, "x": f"{float(z[0]):.6f}", "y": f"{float(z[1]):.6f}"})

def load_adjust():
    p = CFG / "adjust.yml"
    if not p.exists(): return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def load_queries() -> list[str]:
    p = CFG / "queries.txt"
    if not p.exists(): return []
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]
