import argparse, os, json, base64, glob, requests
from pathlib import Path
from typing import List

SYSTEM_PROMPT = (
  "역할: 당신은 동물행동 관찰자입니다. 해석/의미어(위협, 불안 등) 금지. "
  "관찰 가능한 사실만 기록하세요. 아래 스키마로 JSONL만 출력:\n"
  '{"species":"con|cre|unknown","time_range":"mm:ss-mm:ss","body_part":["tail","head","torso","limb"],'
  '"action":["vibrate","still","slow_move","jump","tongue_flick","chirp","rustle"],'
  '"intensity":"low|mid|high|unknown","duration_sec":0.0,'
  '"acoustics":"none|short_chirp|rustle|continuous|unknown",'
  '"context":{"substrate":"paper|cork|glass|unknown","stimulus":"tap|object_show|none|unknown","lighting":"day|night|unknown"},'
  '"notes":""}\n'
  "규칙: 판단어 금지, 불확실하면 unknown, 세그먼트별 1줄, 출력은 JSONL만."
)

def b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")

def collect_images(paths: List[str]) -> List[str]:
    out=[]
    for p in paths:
        out.extend([str(x) for x in Path().glob(p)])
    return [x for x in out if Path(x).is_file()]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[], help="frame_*.jpg 같은 글롭")
    ap.add_argument("--audio", default=None, help="옵션: .wav/.m4a 등 짧은 오디오")
    ap.add_argument("--out", default="slots.jsonl")
    # 환경변수로 LLM 엔드포인트/키/모델 지정
    args=ap.parse_args()
    API_URL   = os.environ.get("LLM_API_URL","")   # 예: https://your-llm/v1/chat/completions
    API_KEY   = os.environ.get("LLM_API_KEY","")
    MODEL     = os.environ.get("LLM_MODEL","multimodal")

    if not API_URL or not API_KEY:
        raise SystemExit("환경변수 LLM_API_URL, LLM_API_KEY가 필요합니다.")

    contents = [{"type":"text","text":"아래 첨부 자료만 근거로 관찰 슬롯을 JSONL로 써주세요."}]
    for img in collect_images(args.images):
        contents.append({"type":"input_image","image":{"base64": b64(img), "mime_type":"image/jpeg"}})
    if args.audio and Path(args.audio).is_file():
        # 일부 API는 오디오를 그대로 받기도 하고, 스펙트로그램 이미지를 요구하기도 합니다.
        contents.append({"type":"input_audio","audio":{"base64": b64(args.audio), "mime_type":"audio/wav"}})

    payload = {
        "model": MODEL,
        "messages": [
            {"role":"system","content":[{"type":"text","text": SYSTEM_PROMPT}]},
            {"role":"user","content": contents}
        ]
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    r = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    # 아래 필드는 제공자에 따라 다릅니다. 필요시 조정하세요.
    result = r.json()
    text = result["choices"][0]["message"]["content"]
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
