#!/usr/bin/env python3
"""Precompute pragmatic vectors from a context file using the plugin model."""
import torch
import json
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
PLUGIN_DIR = os.path.join(ROOT_DIR, "plugin")
MODEL_FILE = os.path.join(PLUGIN_DIR, "model.safetensors")
CTX_PATH = os.path.join(ROOT_DIR, "data", "contexts.txt")
OUT = os.path.join(ROOT_DIR, "vectors", "prag_vectors.pt")
MAX_LEN = 64

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Plugin model not found: {MODEL_FILE}\n"
        "Run: python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('Anni0125/pragmatic-framework', local_dir='plugin')\""
    )

if not os.path.exists(CTX_PATH):
    raise FileNotFoundError(
        f"Contexts file not found: {CTX_PATH}\n"
        "Run: python scripts/make_contexts_from_swda.py first."
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load contexts
contexts=[]
with open(CTX_PATH,"r") as f:
    buf=[]
    for line in f:
        line=line.rstrip()
        if not line:
            if buf:
                contexts.append(" ".join(buf))
                buf=[]
        else:
            buf.append(line)
    if buf:
        contexts.append(" ".join(buf))

print("contexts:", len(contexts))

# tokenizer & encoder
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE).eval()

# load trained head weights
sd = load_file(os.path.join(PLUGIN_DIR,"model.safetensors"))
sd = {k.replace("encoder.",""):v for k,v in sd.items() if k.startswith("encoder.")}
enc.load_state_dict(sd, strict=False)

vecs=[]
with torch.no_grad():
    for ctx in contexts:
        t = tok(ctx, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
        out = enc(**t)
        vec = out.last_hidden_state[:,0,:].squeeze(0).cpu()
        vecs.append(vec)

vecs = torch.stack(vecs)
torch.save(vecs, OUT)
print("saved:", OUT, vecs.shape)
