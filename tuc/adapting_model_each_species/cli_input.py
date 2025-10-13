import argparse, os, yaml, json, uuid
import numpy as np
from tuc.loading_base_model import ingest

def _load_projection(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_vector(q: np.ndarray, meta, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    key = uuid.uuid4().hex[:8]
    np.save(os.path.join(out_dir, f"{key}.npy"), q.astype(np.float32))
    with open(os.path.join(out_dir, f"{key}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return key

def main():
    from ..loading_base_model.ingest import Ingestor
    from ..loading_base_model.encoder import Projector

    p = argparse.ArgumentParser(description="TUC 입력→임베딩 벡터 생성(MVP)")
    sub = p.add_subparsers(dest='cmd', required=True)

    # text
    pt = sub.add_parser('text', help='자연어 텍스트 입력')
    pt.add_argument('--text', required=True, type=str)

    # audio
    ap_audio = sub.add_parser("audio")
    ap_audio.add_argument("--path", required=True)
    ap_audio.add_argument("--out", required=True)
    ap_audio.add_argument("--backend", default=None, choices=[None,"external","wav2vec2"])
    ap_audio.add_argument("--model-id", default=None)
    # video
    pv = sub.add_parser('video', help='비디오 파일 입력')
    pv.add_argument('--path', required=True)

    # sensor
    ps = sub.add_parser('sensor', help='CSV/NPY 센서 배열 입력')
    ps.add_argument('--path', required=True, help='CSV(숫자) 또는 NPY')
    ps.add_argument('--rate', type=int, default=None)

    p.add_argument('--proj', type=str, default='configs/projection.yml')
    p.add_argument('--out', type=str, default='artifacts/inputs')

    args = p.parse_args()

    proj = _load_projection(args.proj)
    ing = Ingestor()
    pj = Projector(proj)

    if args.cmd == 'text':
        payload = ing.from_text(args.text)
        q = pj.text(payload)
        meta = {'kind': 'text', 'text': args.text}

    elif args.cmd == 'audio':
        r = ingest.embed_audio_file(args.path, args.out,
                                    backend=args.backend, model_id=args.model_id)
        print("[cli_input.audio] saved:", r.key, r.meta)
        return

    elif args.cmd == 'video':
        payload = ing.from_video_file(args.path)
        q = pj.video(payload)
        meta = {'kind': 'video', 'path': args.path, 'frames': int(payload.data.shape[0])}

    elif args.cmd == 'sensor':
        # 간단 로더: .npy 또는 .csv(쉼표 기준)
        if args.path.lower().endswith('.npy'):
            arr = np.load(args.path).astype(np.float32)
        else:
            import csv
            rows = []
            with open(args.path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for r in reader:
                    if not r:
                        continue
                    rows.append([float(x) for x in r])
            arr = np.asarray(rows, dtype=np.float32)
        payload = ing.from_sensor_array(arr, rate=args.rate, meta={'src': args.path})
        q = pj.sensor(payload)
        meta = {'kind': 'sensor', 'path': args.path, 'rate': args.rate, 'shape': list(arr.shape)}

    else:
        raise ValueError('unknown cmd')

    key = save_vector(q, meta, args.out)
    print(json.dumps({'key': key, 'out_dir': args.out}, ensure_ascii=False))

if __name__ == '__main__':
    main()
