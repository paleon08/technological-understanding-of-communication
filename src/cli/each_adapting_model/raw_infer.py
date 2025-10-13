import argparse, os, json, csv, numpy as np
from pathlib import Path

def l2n(x, eps=1e-12):
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / n

def load_anchors(npy_path, csv_path):
    A = np.load(npy_path).astype(np.float32)
    meta = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f): meta.append(r)
    assert A.shape[0] == len(meta), "npy/csv 불일치"
    return l2n(A), meta

def load_projection(npz_path):
    if not os.path.isfile(npz_path):
        return {}
    z = np.load(npz_path, allow_pickle=False)
    out = {}
    for k in ("audio","video","sensor"):
        W = z.get(f"{k}.W"); b = z.get(f"{k}.b")
        if W is not None and b is not None:
            out[k] = (W.astype(np.float32), b.astype(np.float32))
    return out

def infer_one(path, proj, A, metaA):
    from tuc.loading_base_model.ingest import Ingestor
    from tuc.loading_base_model.encoder import Projector
    ing, pj = Ingestor(), Projector()
    # 파일 종류 감지
    ext = Path(path).suffix.lower()
    kind = None
    if ext in {".wav",".flac",".m4a",".aac"}:
        payload = ing.from_audio_file(path); z = pj.audio(payload); kind="audio"
    elif ext in {".mp4",".mov",".mkv"}:
        payload = ing.from_video_file(path); z = pj.video(payload); kind="video"
    elif ext in {".npy",".npz",".csv"}:
        # 센서: .npy(또는 .csv 숫자만)로 가정
        if ext == ".npy":
            arr = np.load(path).astype(np.float32)
        else:
            import csv as _csv
            rows=[]
            with open(path,'r',encoding='utf-8') as f:
                for r in _csv.reader(f):
                    if r: rows.append([float(x) for x in r])
            arr = np.asarray(rows, dtype=np.float32)
        payload = ing.from_sensor_array(arr); z = pj.sensor(payload); kind="sensor"
    else:
        raise RuntimeError(f"지원하지 않는 확장자: {ext}")

    # 투영 적용 (없으면 불가)
    if kind not in proj:
        raise RuntimeError(f"projection.npz에 {kind}.W,b가 없습니다. 먼저 train_projection을 돌리세요.")
    W,b = proj[kind]
    q = z @ W + b
    q = l2n(q.astype(np.float32))[0] if q.ndim>1 else l2n(q.astype(np.float32))

    # 코사인 스코어
    scores = (q[None,:] @ A.T)[0]
    idx = np.argsort(-scores)[:5]
    results = []
    for i in idx:
        m = metaA[i]
        results.append({
            "species": m.get("species",""),
            "name": m.get("name",""),
            "score": float(scores[i]),
            "meaning": m.get("meaning",""),
            "tags": m.get("tags",""),
        })
    return kind, results

def main():
    ap = argparse.ArgumentParser(description="원본 파일(영상/오디오/센서)로 바로 판정")
    ap.add_argument("--paths", nargs="+", required=True, help="파일 경로들")
    ap.add_argument("--anchors_npy", default="artifacts/text_anchors/anchors.npy")
    ap.add_argument("--anchors_csv", default="artifacts/text_anchors/anchors.csv")
    ap.add_argument("--projection",  default="configs/projection.npz")
    ap.add_argument("--out", default="artifacts/exp/raw_preds.jsonl")
    args = ap.parse_args()

    A, metaA = load_anchors(args.anchors_npy, args.anchors_csv)
    proj = load_projection(args.projection)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for p in args.paths:
            try:
                kind, res = infer_one(p, proj, A, metaA)
                f.write(json.dumps({"src": p, "kind": kind, "topk": res}, ensure_ascii=False) + "\n")
                print(f"[OK] {p}")
            except Exception as e:
                print(f"[FAIL] {p}: {e}")

if __name__ == "__main__":
    main()
