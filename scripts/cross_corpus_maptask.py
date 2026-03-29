"""
E7d-MapTask: Plugin Predictions vs Gold Labels on MapTask Corpus
Plugin was trained on 293-class MapTask labels; gold uses 13-class coarse labels.
Solution: map both to functional coarse categories for comparison.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file as safe_load
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

PLUGIN_DIR = "../plugin"  # TODO: set to your plugin directory
BASE = "distilbert-base-uncased"
MAX_LEN = 64
BATCH = 256

print("Loading pragmatic plugin...")
with open(os.path.join(PLUGIN_DIR, "label_map.json"), "r") as f:
    m = json.load(f)
lab2id = m["lab2id"]
id2lab = {int(k): v for k, v in m["id2lab"].items()} if isinstance(next(iter(m["id2lab"].keys())), str) else m["id2lab"]
num_labels = len(lab2id)
print(f"Labels: {num_labels}")
print(f"Sample plugin labels: {list(id2lab.values())[:20]}")

tok = AutoTokenizer.from_pretrained(PLUGIN_DIR)

class PragDA(nn.Module):
    def __init__(self, base, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        vec = out.last_hidden_state[:, 0, :]
        return self.classifier(vec), vec

model = PragDA(BASE, num_labels).to(DEVICE).eval()
sd = safe_load(os.path.join(PLUGIN_DIR, "model.safetensors"))
sd = {k.replace("module.", ""): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
print("Model loaded!")

def predict_da(texts):
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            bt = texts[i:i+BATCH]
            batch = tok(bt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
            logits, _ = model(**batch)
            p = logits.argmax(-1).detach().cpu().tolist()
            preds.extend([id2lab[int(x)] for x in p])
    return preds

# Load MapTask utterances
with open("../maptask/maptask_text_da.json") as f:  # TODO: set to your maptask_text_da.json path
    utterances = json.load(f)

texts = [u["text"] for u in utterances]
gold_das = [u["da"] for u in utterances]
print(f"Loaded {len(texts)} MapTask utterances")
print(f"Gold DA types: {sorted(set(gold_das))}")

plugin_preds = predict_da(texts)
plugin_counter = Counter(plugin_preds)
print(f"Plugin prediction types: {len(plugin_counter)}")
print(f"Top-20 plugin predictions: {plugin_counter.most_common(20)}")

# ========================
# Map both to functional coarse categories
# Gold: MapTask 13-class -> coarse
# Plugin: 293-class -> coarse (based on label naming)
# ========================

# Gold 13-class MapTask labels -> functional categories
gold_to_func = {
    "acknowledge": "acknowledge",
    "align": "align",
    "check": "check",
    "clarify": "clarify",
    "explain": "explain",
    "instruct": "instruct",
    "query_yn": "query_yn",
    "query_w": "query_w",
    "ready": "ready",
    "reply_n": "reply_n",
    "reply_w": "reply_w",
    "reply_y": "reply_y",
    "uncodable": "other",
}

# Plugin 293-class labels -> MapTask functional categories
# Based on naming conventions observed in MapTask fine-grained annotations
plugin_to_func = {}
for label in id2lab.values():
    label_lower = label.lower()
    if "acknowledge" in label_lower or label in ["aa", "aap", "aam", "arp", "aar"]:
        plugin_to_func[label] = "acknowledge"
    elif "align" in label_lower:
        plugin_to_func[label] = "align"
    elif "check" in label_lower:
        plugin_to_func[label] = "check"
    elif "clarify" in label_lower:
        plugin_to_func[label] = "clarify"
    elif "explain" in label_lower:
        plugin_to_func[label] = "explain"
    elif "instruct" in label_lower:
        plugin_to_func[label] = "instruct"
    elif "query_yn" in label_lower or "qy" in label_lower:
        plugin_to_func[label] = "query_yn"
    elif "query_w" in label_lower or "qw" in label_lower:
        plugin_to_func[label] = "query_w"
    elif "ready" in label_lower:
        plugin_to_func[label] = "ready"
    elif "reply_n" in label_lower or label in ["nn", "na", "n", "ny"]:
        plugin_to_func[label] = "reply_n"
    elif "reply_w" in label_lower or "qw" in label_lower:
        plugin_to_func[label] = "reply_w"
    elif "reply_y" in label_lower or label in ["yy", "y", "ye", "ym"]:
        plugin_to_func[label] = "reply_y"
    else:
        plugin_to_func[label] = "other"

# Show mapping
print("\nPlugin label -> functional category mapping (top predictions):")
for lab, cnt in plugin_counter.most_common(30):
    func = plugin_to_func.get(lab, "other")
    print(f"  {lab:15s} -> {func} ({cnt})")

# Map to functional
gold_func = [gold_to_func.get(g, "other") for g in gold_das]
plugin_func = [plugin_to_func.get(p, "other") for p in plugin_preds]

gold_func_counter = Counter(gold_func)
plugin_func_counter = Counter(plugin_func)
print(f"\nGold functional distribution: {dict(gold_func_counter.most_common())}")
print(f"Plugin functional distribution: {dict(plugin_func_counter.most_common())}")

# ========================
# E7d-MapTask1: Functional Distribution Comparison
# ========================
all_func_cats = sorted(set(gold_func) | set(plugin_func))
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
gold_sorted = gold_func_counter.most_common(len(gold_func_counter))
das = [d for d, c in gold_sorted]
cnts = [c for d, c in gold_sorted]
ax1.bar(range(len(das)), cnts, color="steelblue")
ax1.set_xticks(range(len(das)))
ax1.set_xticklabels(das, rotation=45, ha="right")
ax1.set_ylabel("Count")
ax1.set_title("E7d-MapTask: Gold Functional Category Distribution")

ax2 = axes[1]
plug_sorted = plugin_func_counter.most_common(len(plugin_func_counter))
das2 = [d for d, c in plug_sorted]
cnts2 = [c for d, c in plug_sorted]
ax2.bar(range(len(das2)), cnts2, color="coral")
ax2.set_xticks(range(len(das2)))
ax2.set_xticklabels(das2, rotation=45, ha="right")
ax2.set_ylabel("Count")
ax2.set_title("E7d-MapTask: Plugin Functional Category Distribution")
plt.tight_layout()
plt.savefig("E7d_maptask1_distribution.png", dpi=150)
print("Saved: E7d_maptask1_distribution.png")

# ========================
# E7d-MapTask2: Functional Confusion Matrix
# ========================
n = len(all_func_cats)
ci = {c: i for i, c in enumerate(all_func_cats)}

cmatrix = np.zeros((n, n))
for gold_c, plug_c in zip(gold_func, plugin_func):
    if gold_c in ci and plug_c in ci:
        cmatrix[ci[gold_c], ci[plug_c]] += 1

row_sums = cmatrix.sum(axis=1, keepdims=True)
cmatrix_norm = cmatrix / (row_sums + 1e-8)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cmatrix_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(all_func_cats, rotation=30, ha="right", fontsize=10)
ax.set_yticklabels(all_func_cats, fontsize=10)
ax.set_xlabel("Plugin Predicted (Functional)")
ax.set_ylabel("Gold (Functional)")
ax.set_title("E7d-MapTask: Confusion Matrix at Functional Category Level\n(Plugin 293-class -> 13 MapTask Gold Classes)")

for i in range(n):
    for j in range(n):
        val = cmatrix_norm[i, j]
        if val > 0.03:
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   color="white" if val > 0.5 else "black", fontsize=9)
plt.colorbar(im, ax=ax, label="Proportion")
plt.tight_layout()
plt.savefig("E7d_maptask2_confusion.png", dpi=150)
print("Saved: E7d_maptask2_confusion.png")

# ========================
# E7d-MapTask3: Per-Category Accuracy
# ========================
per_cat_acc = {}
for cat in gold_func_counter:
    total = sum(1 for g in gold_func if g == cat)
    correct = sum(1 for g, p in zip(gold_func, plugin_func) if g == cat and p == cat)
    per_cat_acc[cat] = correct / total if total > 0 else 0

cats = [d for d, c in gold_func_counter.most_common()]
accs = [per_cat_acc[d] for d in cats]

fig, ax = plt.subplots(figsize=(12, 5))
colors = ["green" if a > 0.6 else "orange" if a > 0.3 else "red" for a in accs]
ax.bar(range(len(cats)), accs, color=colors)
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cats, rotation=30, ha="right")
ax.set_ylabel("Plugin Accuracy")
ax.set_ylim(0, 1.15)
ax.axhline(y=0.6, color="green", linestyle="--", alpha=0.5, label="good (0.6)")
ax.axhline(y=0.3, color="orange", linestyle="--", alpha=0.5, label="fair (0.3)")
ax.set_title("E7d-MapTask: Per-Functional-Category Plugin Accuracy")
ax.legend()
for i, (d, a) in enumerate(zip(cats, accs)):
    ax.text(i, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig("E7d_maptask3_per_cat_acc.png", dpi=150)
print("Saved: E7d_maptask3_per_cat_acc.png")

# ========================
# Summary
# ========================
total_correct = sum(1 for g, p in zip(gold_func, plugin_func) if g == p)
overall_acc = total_correct / len(gold_func)

print(f"\n=== E7d-MapTask Summary ===")
print(f"Total utterances: {len(gold_func)}")
print(f"Overall functional accuracy: {overall_acc:.3f} ({total_correct}/{len(gold_func)})")
print(f"\nPer-category accuracy (functional):")
for cat, cnt in gold_func_counter.most_common():
    acc = per_cat_acc[cat]
    print(f"  {cat:12s}: {acc:.3f} ({cnt} samples)")

print("\nE7d-MapTask complete!")
