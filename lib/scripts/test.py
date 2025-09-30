from transformers import ClapProcessor, ClapModel
import torch

MODEL_NAME = "laion/clap-htsat-unfused"

processor = ClapProcessor.from_pretrained(MODEL_NAME)
model = ClapModel.from_pretrained(MODEL_NAME)
model.eval()

with torch.no_grad():
    text_inputs = processor(text=["hiss"], return_tensors="pt", padding=True)
    text_feat = model.get_text_features(**text_inputs)
    print("Text embedding shape:", tuple(text_feat.shape))
