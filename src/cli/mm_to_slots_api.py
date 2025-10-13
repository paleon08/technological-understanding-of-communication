import argparse, os, json, base64, requests
from pathlib import Path

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

def main():
    ap=argparse.ArgumentParser(description="멀티모달 입력 -> 관찰 슬롯 JSONL")
    ap.add_argument("--frames", nargs="*", default=[], help="이미지 경로들 (jpg/png)")
    ap.add_argument("--audio", default=None, help="선택: 짧은 오디오(.wav/.m4a 등)")
    ap.add_argument("--sensor_img", nargs="*", default=[], help="선택: 센서 시각화 이미지들")
    ap.add_argument("--out", default="slots.jsonl")
    args=ap.parse_args()

    API_URL = os.environ.get("LLM_API_URL","")
    API_KEY = os.environ.get("LLM_API_KEY","")
    MODEL   = os.environ.get("LLM_MODEL","multimodal")
    if not API_URL or not API_KEY:
        raise SystemExit("환경변수 LLM_API_URL / LLM_API_KEY 필요")

    # 멀티모달 content 구성 (벤더별 포맷만 맞게 조정)
    content = [{"type":"text","text":"아래 첨부 자료만 근거로 관찰 슬롯을 JSONL로 출력하세요."}]
    for p in (args.frames or []):
        content.append({"type":"input_image","image":{"base64": b64(p), "mime_type":"image/jpeg"}})
    if args.audio and Path(args.audio).is_file():
        content.append({"type":"input_audio","audio":{"base64": b64(args.audio), "mime_type":"audio/wav"}})
    for p in (args.sensor_img or []):
        content.append({"type":"input_image","image":{"base64": b64(p), "mime_type":"image/png"}})

    payload = {
        "model": MODEL,
        "messages": [
          {"role":"system","content":[{"type":"text","text": SYSTEM_PROMPT}]},
          {"role":"user","content": content}
        ]
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    r = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=90)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    # JSONL만 추출 (혹시 설명 섞이면 줄별 {…}만 필터)
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("{")]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {args.out} ({len(lines)} lines)")

if __name__ == "__main__":
    main()
