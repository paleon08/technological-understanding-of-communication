# src/cli/model_loading/build_subspace.py
from __future__ import annotations
import os, argparse, json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import yaml

# local imports
from tuc.loading_base_model.encoder import encode_text
from tuc.loading_base_model.subspace import fit_subspace, SubspaceProjector

def _load_prompts(yaml_path: str, meaning_only: bool = True) -> List[Dict[str, Any]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    anchors = data.get("anchors", [])
    out = []
    for a in anchors:
        name = a.get("name", "")
        prompt = a.get("prompt") or a.get("text") or a.get("desc") or ""
        if not prompt:
            continue
        if meaning_only and not str(name).startswith("meaning."):
            continue
        out.append({"name": name, "prompt": prompt})
    if not out:
        raise ValueError("No prompts found. Check anchors and meaning_only flag.")
    return out

def main():
    ap = argparse.ArgumentParser("build_subspace")
    ap.add_argument("--anchors-yaml", required=True, help="YAML with anchors[].name/prompt")
    ap.add_argument("--out", default="artifacts/subspace/canon_subspace.npz")
    ap.add_argument("--meaning-only", default="true", choices=["true","false"])
    ap.add_argument("--center", default="true", choices=["true","false"])
    ap.add_argument("--var-thresh", type=float, default=0.95, help="explained variance threshold")
    ap.add_argument("--max-dim", type=int, default=32, help="hard cap for k (optional)")
    ap.add_argument("--mode", default="passage", choices=["query","passage"], help="E5 prefix mode for prompts")
    args = ap.parse_args()

    prompts = _load_prompts(args.anchors_yaml, meaning_only=(args.meaning_only=="true"))
    texts = [p["prompt"] for p in prompts]

    # Embed prompts (same encoder as runtime)
    X = encode_text(texts, mode=args.mode)   # [m, d] (already L2 normalized)
    B, mu, k, stats = fit_subspace(
        anchors=X,
        center=(args.center=="true"),
        var_thresh=args.var_thresh,
        max_dim=args.max_dim
    )

    proj = SubspaceProjector(
        B=B, mu=mu, center=(args.center=="true"),
        meta={
            "source_yaml": os.path.abspath(args.anchors_yaml),
            "meaning_only": (args.meaning_only=="true"),
            "var_thresh": args.var_thresh,
            "k": k,
            "explained_var": stats["explained_var"],
            "encoder_env": {
                "TUC_TEXT_MODEL": os.getenv("TUC_TEXT_MODEL", ""),
                "TUC_E5_MODE": os.getenv("TUC_E5_MODE", ""),
                "TUC_NORMALIZE": os.getenv("TUC_NORMALIZE", ""),
            }
        }
    )
    out_path = Path(args.out)
    proj.save(str(out_path))
    print(json.dumps({
        "out_npz": str(out_path),
        "k": k,
        "explained_var": stats["explained_var"],
        "m_prompts": int(X.shape[0]),
        "d": int(X.shape[1]),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
