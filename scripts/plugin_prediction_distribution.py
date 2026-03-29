"""
Supplementary Experiments for LRE Paper
E7d: Confusion Matrix
E7e: Pragmatic Space Interpolation
Human vs Plugin DA Comparison
Multi-label Analysis
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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

PLUGIN_DIR = "../plugin"  # TODO: set to your plugin directory
BASE = "distilbert-base-uncased"
MAX_LEN = 64
BATCH = 128
N_SAMPLE_LARGE = 5000

print("Loading pragmatic plugin...")
with open(os.path.join(PLUGIN_DIR, "label_map.json"), "r") as f:
    m = json.load(f)
lab2id = m["lab2id"]
id2lab = {int(k): v for k, v in m["id2lab"].items()} if isinstance(next(iter(m["id2lab"].keys())), str) else m["id2lab"]
num_labels = len(lab2id)
print(f"Labels: {num_labels}")

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

def predict_da_vectors(texts):
    preds = []
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            bt = texts[i:i+BATCH]
            batch = tok(bt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
            logits, vec = model(**batch)
            p = logits.argmax(-1).detach().cpu().tolist()
            preds.extend([id2lab[int(x)] for x in p])
            vecs.extend(vec.cpu().numpy())
    return preds, np.array(vecs)

print("\n=== E7d: Confusion Matrix ===")
sw_texts = []
sw_true_das = []
with open("../data/test.jsonl") as f:  # TODO: set to your data directory
    for i, line in enumerate(f):
        if i >= N_SAMPLE_LARGE:
            break
        d = json.loads(line)
        sw_texts.append(d["text"])
        sw_true_das.append(d.get("da", d.get("label", "unknown")))

print(f"Loaded {len(sw_texts)} SwDA utterances")
sample = json.loads(open("../data/test.jsonl").readline())
print(f"Sample keys: {sample.keys()}")
print(f"Sample text: {sample.get('text', 'N/A')[:80]}")
print(f"Sample DA: {sample.get('da', sample.get('label', 'N/A'))}")

sw_plugin_preds, sw_vecs = predict_da_vectors(sw_texts)
print(f"Plugin predictions: {len(sw_plugin_preds)}")

true_counter = Counter(sw_true_das)
plugin_counter = Counter(sw_plugin_preds)
top_true_das = [d for d, c in true_counter.most_common(15)]
top_plugin_das = [d for d, c in plugin_counter.most_common(15)]
print(f"Top true DAs: {top_true_das}")
print(f"Top plugin DAs: {top_plugin_das}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax1 = axes[0]
top_true = true_counter.most_common(12)
das = [d for d, c in top_true]
cnts = [c for d, c in top_true]
ax1.bar(range(len(das)), cnts, color="steelblue")
ax1.set_xticks(range(len(das)))
ax1.set_xticklabels(das, rotation=30, ha="right")
ax1.set_ylabel("Count")
ax1.set_title("E7d: True DA Distribution (SwDA Human Annotation)")
ax2 = axes[1]
top_plug = plugin_counter.most_common(12)
das2 = [d for d, c in top_plug]
cnts2 = [c for d, c in top_plug]
ax2.bar(range(len(das2)), cnts2, color="coral")
ax2.set_xticks(range(len(das2)))
ax2.set_xticklabels(das2, rotation=30, ha="right")
ax2.set_ylabel("Count")
ax2.set_title("E7d: Plugin Prediction Distribution (293-class Plugin)")
plt.tight_layout()
plt.savefig("E7d1_da_comparison.png", dpi=150)
print("Saved: E7d1_da_comparison.png")

top_n = 10
top_true_labels = [d for d, c in true_counter.most_common(top_n)]
top_plugin_labels = [d for d, c in plugin_counter.most_common(top_n)]
cmatrix = np.zeros((top_n, top_n))
for true_d, pred_d in zip(sw_true_das, sw_plugin_preds):
    if true_d in top_true_labels and pred_d in top_plugin_labels:
        i = top_true_labels.index(true_d)
        j = top_plugin_labels.index(pred_d)
        cmatrix[i, j] += 1
row_sums = cmatrix.sum(axis=1, keepdims=True)
cmatrix_norm = cmatrix / (row_sums + 1e-8)

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cmatrix_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(top_n))
ax.set_yticks(range(top_n))
ax.set_xticklabels(top_plugin_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(top_true_labels, fontsize=9)
ax.set_xlabel("Plugin Predicted DA")
ax.set_ylabel("True DA (Human Annotation)")
ax.set_title("E7d: Confusion Matrix - Plugin Predictions vs Human Labels\n(Row-normalized)")
for i in range(top_n):
    for j in range(top_n):
        val = cmatrix_norm[i, j]
        if val > 0.05:
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   color="white" if val > 0.5 else "black", fontsize=8)
plt.colorbar(im, ax=ax, label="Proportion")
plt.tight_layout()
plt.savefig("E7d2_confusion_matrix.png", dpi=150)
print("Saved: E7d2_confusion_matrix.png")

print("\nPer-true-DA accuracy (top-10):")
for i, da in enumerate(top_true_labels):
    total = cmatrix[i].sum()
    if total > 0:
        acc = cmatrix[i, i] / total
        print(f"  {da}: {acc:.3f} ({int(total)} samples)")

print("\n=== E7e: Pragmatic Space Interpolation ===")
da_groups = defaultdict(list)
for da, vec in zip(sw_plugin_preds, sw_vecs):
    da_groups[da].append(vec)

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

top_das_for_interp = ["sd", "b", "sv", "qy", "aa"]
available_das = [d for d in top_das_for_interp if d in da_groups and len(da_groups[d]) > 10]
print(f"Available DAs for interpolation: {available_das}")

mean_vecs = {}
for da in available_das:
    vecs = np.array(da_groups[da])
    mean_vecs[da] = np.mean(vecs, axis=0)

if len(available_das) >= 2:
    da1, da2 = available_das[0], available_das[1]
    v1, v2 = mean_vecs[da1], mean_vecs[da2]
    alphas = np.linspace(0, 1, 11)
    interp_vecs = [v1 * (1 - a) + v2 * a for a in alphas]

    all_mean_vecs = np.array([mean_vecs[d] for d in available_das])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_mean_vecs)
    interp_2d = pca.transform(interp_vecs)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(available_das)))
    for i, da in enumerate(available_das):
        ax.scatter(all_2d[i, 0], all_2d[i, 1], s=200, c=[colors[i]], zorder=5, label=f"{da} (mean)")
        ax.annotate(da, (all_2d[i, 0]+0.05, all_2d[i, 1]+0.05), fontsize=11, fontweight="bold")
    ax.plot(interp_2d[:, 0], interp_2d[:, 1], "k--", alpha=0.5, linewidth=2, label="Interpolation path")
    ax.scatter(interp_2d[0, 0], interp_2d[0, 1], c="green", s=200, marker="*", zorder=7, label=f"{da1} (start)")
    ax.scatter(interp_2d[-1, 0], interp_2d[-1, 1], c="red", s=200, marker="*", zorder=7, label=f"{da2} (end)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"E7e: Pragmatic Space Interpolation\n({da1} -> {da2})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("E7e_interpolation.png", dpi=150)
    print("Saved: E7e_interpolation.png")

    print("\nCosine similarity along interpolation path:")
    for j, a in enumerate(alphas):
        sim_to_v1 = cosine_similarity([interp_vecs[j]], [v1])[0, 0]
        sim_to_v2 = cosine_similarity([interp_vecs[j]], [v2])[0, 0]
        print(f"  alpha={a:.1f}: sim_to_{da1}={sim_to_v1:.3f}, sim_to_{da2}={sim_to_v2:.3f}")

    print("\nNearest DA type along interpolation path:")
    for j, a in enumerate(alphas):
        best_da = None
        best_sim = -1
        for da in available_das:
            if da == da1 or da == da2:
                continue
            sim = cosine_similarity([interp_vecs[j]], [mean_vecs[da]])[0, 0]
            if sim > best_sim:
                best_sim = sim
                best_da = da
        if j in [0, 5, 10]:
            print(f"  alpha={a:.1f}: nearest={best_da} (sim={best_sim:.3f})")

print("\n=== Human vs Plugin DA Comparison ===")
common_das = [d for d, c in true_counter.most_common(10) if d in plugin_counter]
true_total = sum(true_counter.values())
plugin_total = len(sw_plugin_preds)

if len(common_das) > 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(common_das))
    w = 0.35
    true_props = [true_counter[d] / true_total for d in common_das]
    plugin_props = [plugin_counter.get(d, 0) / plugin_total for d in common_das]
    ax.bar(x - w/2, true_props, w, label="True (Human)", color="steelblue")
    ax.bar(x + w/2, plugin_props, w, label="Plugin Predicted", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(common_das, rotation=30, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("Human vs Plugin: DA Distribution Comparison\n(Top DAs with Matching Labels)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("E7d3_human_vs_plugin.png", dpi=150)
    print("Saved: E7d3_human_vs_plugin.png")
else:
    print("No matching DAs between human annotations and plugin predictions - skipping comparison chart")

print("\n=== Multi-label Analysis ===")
multi_label_count = 0
multi_label_examples = []
with open("../data/test.jsonl") as f:  # TODO: set to your data directory
    for i, line in enumerate(f):
        if i >= N_SAMPLE_LARGE:
            break
        d = json.loads(line)
        da_val = d.get("da", d.get("label", ""))
        if isinstance(da_val, list):
            multi_label_count += 1
            if len(multi_label_examples) < 5:
                multi_label_examples.append(d)
print(f"Multi-label utterances: {multi_label_count} / {N_SAMPLE_LARGE}")
print(f"Sample multi-label examples: {multi_label_examples[:3]}")

sample_da_field = None
with open("../data/test.jsonl") as f:  # TODO: set to your data directory
    d = json.loads(f.readline())
    sample_da_field = d.get("da", d.get("label", "unknown"))
    print(f"Sample DA field type: {type(sample_da_field)}, value: {sample_da_field}")

da_cooccur = defaultdict(Counter)
prev_da = None
with open("../data/test.jsonl") as f:  # TODO: set to your data directory
    for i, line in enumerate(f):
        if i >= N_SAMPLE_LARGE:
            break
        d = json.loads(line)
        curr_da = d.get("da", d.get("label", "unknown"))
        if prev_da is not None:
            da_cooccur[prev_da][curr_da] += 1
        prev_da = curr_da

print("\nDA transition co-occurrence analysis:")
transitions = []
for prev_da, next_das in da_cooccur.items():
    for next_da, count in next_das.items():
        transitions.append((prev_da, next_da, count))
transitions.sort(key=lambda x: -x[2])
for prev_da, next_da, count in transitions[:10]:
    print(f"  {prev_da} -> {next_da}: {count}")

print("\n=== E7 Supplementary Complete ===")
