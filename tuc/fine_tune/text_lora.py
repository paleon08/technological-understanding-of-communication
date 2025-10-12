# tuc/fine_tune/text_lora.py (요지)
from __future__ import annotations
import os, json, random, torch
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

@dataclass
class TrainConf:
    model_id: str = os.getenv("TUC_TEXT_MODEL","laion/clap-htsat-fused")
    out_dir: str = "artifacts/models/text_adapter"
    lr: float = 1e-4
    epochs: int = 3
    batch_size: int = 16
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 7

# --- 데이터 구성: anchors/rules에서 문장 쌍(양/음) 만들기 ---
class PairDataset(Dataset):
    def __init__(self, pos_pairs, neg_pairs, tokenizer):
        self.pos = pos_pairs; self.neg = neg_pairs; self.tokenizer = tokenizer
    def __len__(self): return len(self.pos) + len(self.neg)
    def __getitem__(self, i):
        # label: 1=positive, 0=negative
        # 샘플링 단순화(데모)
        if i < len(self.pos):
            a,b = self.pos[i]; y=1
        else:
            a,b = self.neg[i-len(self.pos)]; y=0
        return {"a": a, "b": b, "y": y}

def collate(batch, tokenizer):
    A = [x["a"] for x in batch]; B = [x["b"] for x in batch]
    tokA = tokenizer(A, padding=True, truncation=True, return_tensors="pt")
    tokB = tokenizer(B, padding=True, truncation=True, return_tensors="pt")
    y = torch.tensor([x["y"] for x in batch], dtype=torch.float32)
    return tokA, tokB, y

def build_pairs_from_configs():
    # TODO: anchors/*.yml, rules/*.yml 파싱해서
    # pos_pairs = [(text_i, text_j), ...], neg_pairs = [...]
    # 의미 채널 기준으로 묶기/분리 적용
    return [("territorial warning", "aggressive dog bark used as an alarm call")], \
           [("play invitation", "territorial warning")]

def train(conf: TrainConf):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(conf.model_id)
    base = AutoModel.from_pretrained(conf.model_id).to(device)
    lora = LoraConfig(r=conf.lora_r, lora_alpha=conf.lora_alpha, lora_dropout=conf.lora_dropout, bias="none", target_modules=["q_proj","v_proj"])
    model = get_peft_model(base, lora).train()

    pos_pairs, neg_pairs = build_pairs_from_configs()
    ds = PairDataset(pos_pairs, neg_pairs, tok)
    dl = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tok))

    opt = torch.optim.AdamW(model.parameters(), lr=conf.lr)
    for ep in range(conf.epochs):
        for tokA, tokB, y in dl:
            tokA, tokB, y = {k:v.to(device) for k,v in tokA.items()}, {k:v.to(device) for k,v in tokB.items()}, y.to(device)
            outA = model(**tokA); outB = model(**tokB)
            va = outA.last_hidden_state.mean(dim=1); vb = outB.last_hidden_state.mean(dim=1)
            va = torch.nn.functional.normalize(va, dim=-1); vb = torch.nn.functional.normalize(vb, dim=-1)
            # 간단한 contrastive loss(코사인 마진)
            cos = (va*vb).sum(dim=-1)
            loss = (y*(1-cos) + (1-y)*torch.clamp(cos-0.3, min=0)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[text_lora] epoch {ep+1} loss={loss.item():.4f}")

    os.makedirs(conf.out_dir, exist_ok=True)
    model.save_pretrained(conf.out_dir)
    print("[text_lora] saved adapter to", conf.out_dir)

if __name__ == "__main__":
    train(TrainConf())
