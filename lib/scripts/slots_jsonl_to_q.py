import argparse, json, uuid, numpy as np
from pathlib import Path

def render_slot(s):
    sp = s.get("species","unknown")
    bp = ",".join(s.get("body_part",[]) or [])
    act = ",".join(s.get("action",[]) or [])
    inten = s.get("intensity","unknown")
    dur = s.get("duration_sec", 0)
    ac = s.get("acoustics","unknown")
    ctx = s.get("context",{}) or {}
    ctx_str = ";".join(f"{k}={v}" for k,v in ctx.items())
    return f"species {sp}; body {bp}; action {act}; intensity {inten}; duration {dur}s; acoustics {ac}; context {ctx_str}"

def main():
    ap=argparse.ArgumentParser(description="관찰 슬롯 JSONL -> q 벡터 저장")
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_dir", default="artifacts/inputs")
    args=ap.parse_args()

    from tuc.encoder import Projector
    pj = Projector()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    n=0
    with open(args.in_jsonl,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            s = json.loads(line)
            text = render_slot(s)
            # 텍스트 임베딩 경로로 q 생성
            class P: pass
            p=P(); p.data=text
            q = pj.text(p)  # [D]
            key = f"slot_{s.get('species','unk')}_{uuid.uuid4().hex[:8]}"
            np.save(out/f"{key}.npy", q.astype(np.float32))
            (out/f"{key}.json").write_text(
                json.dumps({"kind":"text","src":args.in_jsonl,"slot":s,"rendered":text}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            n+=1
    print(f"[OK] wrote {n} vectors to {str(out)}")

if __name__=="__main__":
    main()
