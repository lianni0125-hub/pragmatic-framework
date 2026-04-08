#!/usr/bin/env python3
"""Tag user data with DA predictions using the plugin."""
import os, json, torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file as safe_load
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PLUGIN_DIR = os.path.join(ROOT_DIR, "plugin")
MODEL_FILE = os.path.join(PLUGIN_DIR, "model.safetensors")
IN_PATH  = os.path.join(ROOT_DIR, "data", "gpt2_before_after.jsonl")
OUT_PATH = os.path.join(ROOT_DIR, "data", "gpt2_before_after_tagged.jsonl")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Plugin model not found: {MODEL_FILE}\n"
        "Run: python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('Anni0125/pragmatic-framework', local_dir='plugin')\""
    )

if not os.path.exists(IN_PATH):
    raise FileNotFoundError(
        f"Input file not found: {IN_PATH}\n"
        "This script requires your own data file at data/gpt2_before_after.jsonl.\n"
        "Each line should be JSON: {\"baseline\": \"...\", \"prag\": \"...\"}"
    )
MAX_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = "distilbert-base-uncased"
BATCH = 128

with open(os.path.join(PLUGIN_DIR, "label_map.json"), "r") as f:
    m = json.load(f)
lab2id = m["lab2id"]
id2lab = {int(k):v for k,v in m["id2lab"].items()} if isinstance(next(iter(m["id2lab"].keys())), str) else m["id2lab"]
num_labels = len(lab2id)

tok = AutoTokenizer.from_pretrained(PLUGIN_DIR)

class PragDA(nn.Module):
    def __init__(self, base, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        vec = out.last_hidden_state[:,0,:]
        return self.classifier(vec)

model = PragDA(BASE, num_labels).to(DEVICE).eval()
sd = safe_load(os.path.join(PLUGIN_DIR, "model.safetensors"))
sd = {k.replace("module.",""): v for k,v in sd.items()}
model.load_state_dict(sd, strict=False)

def predict_da(texts):
    preds=[]
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            bt = texts[i:i+BATCH]
            batch = tok(bt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
            logits = model(**batch)
            p = logits.argmax(-1).detach().cpu().tolist()
            preds.extend([id2lab[int(x)] for x in p])
    return preds

rows=[]
with open(IN_PATH,"r") as f:
    for line in f:
        rows.append(json.loads(line))

base_texts = [r["baseline"] for r in rows]
prag_texts = [r["prag"] for r in rows]

base_da = predict_da(base_texts)
prag_da = predict_da(prag_texts)

with open(OUT_PATH,"w") as w:
    for r, b, p in zip(rows, base_da, prag_da):
        r["baseline_da"] = b
        r["prag_da"] = p
        w.write(json.dumps(r, ensure_ascii=False) + "\n")

print("saved:", OUT_PATH)
print("baseline DA top:", Counter(base_da).most_common(10))
print("prag DA top:", Counter(prag_da).most_common(10))
