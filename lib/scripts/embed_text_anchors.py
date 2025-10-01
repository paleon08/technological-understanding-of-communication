import os, sys, glob, yaml, json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import ClapProcessor, ClapModel
import torch

# ===== 설정 =====
ANCHOR_DIR = "configs/anchors"
OUT_DIR = "artifacts/text_anchors"
MODEL_NAME = "laion/clap-htsat-unfused"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# ===== 모델 로딩 =====
print(f"[model] loading {MODEL_NAME}")
processor = ClapProcessor.from_pretrained(MODEL_NAME)
model = ClapModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ===== YML 수집 =====
files = sorted(glob.glob(os.path.join(ANCHOR_DIR, "*.yml")))
if not files:
    print(f"[warn] no anchor files found in {ANCHOR_DIR}")
    sys.exit(0)

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        yml = yaml.safe_load(f)

    name = yml.get("name") or Path(path).stem
    anchors = yml.get("anchors") or []
    if not anchors:
        print(f"[skip] {name}: no anchors")
        continue

    print(f"[embed] {name} ({len(anchors)} anchors)")
    with torch.no_grad():
        inputs = processor(text=anchors, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        emb = model.get_text_features(**inputs)
        emb = torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()

    # 저장
    meta = pd.DataFrame({"index": list(range(len(anchors))), "text": anchors})
    meta.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False)
    np.save(os.path.join(OUT_DIR, f"{name}.npy"), emb)
    print(f"  -> saved {name}.npy / {name}.csv")

print(f"[done] all anchors embedded to: {OUT_DIR}")
