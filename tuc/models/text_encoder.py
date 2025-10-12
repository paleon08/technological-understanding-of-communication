# tuc/models/text_encoder.py
from __future__ import annotations
import os, torch
from transformers import AutoTokenizer, AutoModel
try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

DEFAULT_TEXT_MODEL = os.getenv("TUC_TEXT_MODEL", "laion/clap-htsat-fused")
DEFAULT_TEXT_ADAPTER = os.getenv("TUC_TEXT_ADAPTER", "artifacts/models/text_adapter")  # 자동 감지 경로

class TextEncoder:
    def __init__(self, model_id: str|None=None, device: str|None=None, adapter_dir: str|None=None):
        self.model_id = model_id or DEFAULT_TEXT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        # 어댑터 자동 로드
        use_adapter = adapter_dir or (os.path.isdir(DEFAULT_TEXT_ADAPTER) and DEFAULT_TEXT_ADAPTER)
        if use_adapter and HAS_PEFT:
            self.model = PeftModel.from_pretrained(self.model, use_adapter).to(self.device).eval()
            print(f"[TextEncoder] adapter loaded: {use_adapter}")

    @torch.no_grad()
    def encode(self, texts: list[str]):
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.model(**toks)
        # CLAP/Text 계열이면 get_text_features가 따로 있을 수 있음 → 안전하게 평균 풀링
        last = getattr(out, "last_hidden_state", None)
        if last is None:
            # 모델별 분기 추가 가능
            raise RuntimeError("No last_hidden_state; add model-specific pooling here.")
        vec = last.mean(dim=1)                       # [B, D]
        vec = torch.nn.functional.normalize(vec, dim=-1)
        return vec.detach().cpu().numpy()
