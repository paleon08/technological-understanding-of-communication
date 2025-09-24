import os, json, math, argparse
from pathlib import Path
import numpy as np
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

# ----- config -----
cfg = yaml.safe_load(open("configs/download.yaml","r",encoding="utf-8"))
META = Path(cfg["paths"]["meta"])
PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
SR = int(cfg["audio"]["target_sr"])
LABELS = cfg["labels"]
L2I = {l:i for i,l in enumerate(LABELS)}
NCLASS = len(LABELS)

# ----- data -----
class AudioJsonl(Dataset):
    def __init__(self, jsonl, split=None, sec=10.0):
        self.items=[]
        with open(jsonl,"r",encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if split and ex["split"]!=split: continue
                self.items.append(ex)
        self.sec = sec
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=320, win_length=1024,
            n_mels=80, f_min=50, f_max=SR//2
        )
        self.amp2db = torchaudio.transforms.AmplitudeToDB()
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        ex = self.items[i]
        wav, sr = torchaudio.load(ex["audio_path"])
        if sr!=SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        # ensure 10s mono
        wav = wav.mean(0, keepdim=True)
        need = int(SR*self.sec)
        if wav.shape[1] < need:
            pad = need - wav.shape[1]
            wav = F.pad(wav, (0,pad))
        else:
            wav = wav[:,:need]
        # spec
        mel = self.amp2db(self.mel(wav))  # [1,80,T]
        mel = (mel - mel.mean())/(mel.std()+1e-8)
        y = torch.zeros(NCLASS)
        for l in ex["labels"]:
            if l in L2I: y[L2I[l]] = 1.0
        return mel, y, ex["id"]

# ----- model (tiny CRNN) -----
class CRNN(nn.Module):
    def __init__(self, n_mels=80, emb_dim=512, n_class=6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d((2,2))
        )
        self.gru = nn.GRU(input_size=128*(n_mels//8), hidden_size=emb_dim//2,
                          num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(emb_dim, n_class)
        self.emb_dim = emb_dim
    def forward(self, x):
        # x: [B,1,80,T]
        B,_,M,T = x.shape
        h = self.cnn(x)               # [B,128, M/8, T/8]
        h = h.permute(0,3,1,2).contiguous()  # [B,T/8,128,M/8]
        h = h.view(B, h.shape[1], -1)        # [B,T/8, 128*(M/8)]
        out, _ = self.gru(h)                 # [B,T/8, emb]
        emb = out.mean(1)                    # [B, emb]
        logit = self.head(emb)               # [B, n_class]
        return logit, emb

def train_one_epoch(model, loader, opt, device):
    model.train()
    loss_meter=0.0
    for mel, y, _ in loader:
        mel, y = mel.to(device), y.to(device)
        logit, _ = model(mel)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_meter += loss.item()*mel.size(0)
    return loss_meter/len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true=[]; y_pred=[]
    for mel, y, _ in loader:
        mel = mel.to(device)
        logit, _ = model(mel)
        y_true.append(y.numpy())
        y_pred.append(torch.sigmoid(logit).cpu().numpy())
    y_true = np.concatenate(y_true); y_pred=np.concatenate(y_pred)
    # macro-F1@0.5 threshold
    th=0.5
    tp = ((y_pred>=th)&(y_true==1)).sum(axis=0)
    fp = ((y_pred>=th)&(y_true==0)).sum(axis=0)
    fn = ((y_pred< th)&(y_true==1)).sum(axis=0)
    f1 = (2*tp / (2*tp+fp+fn+1e-8)).mean()
    return float(f1)

@torch.no_grad()
def dump_embeddings(model, loader, device, out_prefix):
    model.eval()
    all_emb=[]; all_id=[]
    for mel, _, ids in loader:
        mel = mel.to(device)
        _, emb = model(mel)
        all_emb.append(emb.cpu().numpy())
        all_id.extend(ids)
    E = np.concatenate(all_emb)
    np.save(f"{out_prefix}_emb.npy", E)
    with open(f"{out_prefix}_ids.txt","w",encoding="utf-8") as f:
        f.write("\n".join(all_id))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    jsonl = META/"manifest.jsonl"
    train_ds = AudioJsonl(jsonl, "train")
    val_ds   = AudioJsonl(jsonl, "val")
    test_ds  = AudioJsonl(jsonl, "test")
    tr = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    va = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=0)
    te = DataLoader(test_ds,  batch_size=args.bs, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNN(n_mels=80, emb_dim=512, n_class=NCLASS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best= -1
    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, tr, opt, device)
        f1 = evaluate(model, va, device)
        print(f"[{ep:02d}] loss={tr_loss:.4f} valF1={f1:.3f}")
        if f1>best:
            best=f1
            torch.save(model.state_dict(), MODEL_DIR/"audio_baseline.pt")

    # 임베딩 덤프(전이/UMAP용)
    model.load_state_dict(torch.load(MODEL_DIR/"audio_baseline.pt", map_location=device))
    dump_embeddings(model, tr, device, str(PROC/"train"))
    dump_embeddings(model, va, device, str(PROC/"val"))
    dump_embeddings(model, te, device, str(PROC/"test"))
    print("[OK] saved:", MODEL_DIR/"audio_baseline.pt")

if __name__=="__main__":
    main()
