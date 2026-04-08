#!/usr/bin/env python3
"""Analyze speaker profiles from SwDA zip transcripts."""
import json, numpy as np, zipfile, csv, io
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch

SWDA_ZIP = os.path.join(os.path.dirname(__file__), "..", "swda.zip")
if not os.path.exists(SWDA_ZIP):
    raise FileNotFoundError(
        f"SwDA zip file not found: {SWDA_ZIP}\n"
        "Please download the Switchboard-I transcripts from:\n"
        "  https://www.switchboard.co.uk/#data\n"
        "Place the downloaded swda.zip in the project root directory."
    )

print("=== Extracting speaker info from swda.zip ===")
# Build conv_id -> {utterance_index -> caller} mapping
# Only for conversations that appear in our data
utt_speaker = {}  # conv_id -> utt_idx -> caller
target_convs = set()

# First, collect all conv_ids from splits_final2
for split in ["train", "valid", "test"]:
    path = f"../data/{split}.jsonl"
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            target_convs.add(d["conv_id"])

print(f"Target conversations: {len(target_convs)}")

z = zipfile.ZipFile(SWDA_ZIP)
utt_files = [n for n in z.namelist() if n.endswith('.utt.csv') and '__MACOSX' not in n]
print(f"Total .utt.csv files in zip: {len(utt_files)}")

files_checked = 0
matched_files = 0

for fname in utt_files:
    # Extract conversation number from filename like "swda/sw00utt/sw_0001_4325.utt.csv"
    # -> conv_id = "sw_0001_4325"
    parts = fname.split('/')
    if len(parts) < 2:
        continue
    fname_only = parts[-1]  # sw_0001_4325.utt.csv
    # conv_id from filename: sw_0001_4325
    m = fname_only.split('_')
    if len(m) < 2:
        continue
    conv_no = fname_only.replace('sw_', '').replace('.utt.csv', '').split('_')[-1]
    # Actually the format is sw_NNNN_NNNN -> conv_id = sw_NNNN_NNNN
    conv_id = 'sw_' + fname_only.replace('.utt.csv', '').replace('sw_', '')
    if conv_id not in target_convs:
        files_checked += 1
        continue
    matched_files += 1

    content = z.read(fname).decode('utf-8', errors='replace')
    reader = csv.reader(io.StringIO(content))
    header = next(reader)
    try:
        caller_idx = header.index('caller')
        utt_idx_idx = header.index('utterance_index')
    except ValueError:
        continue

    utt_speaker[conv_id] = {}
    for row in reader:
        if len(row) <= max(caller_idx, utt_idx_idx):
            continue
        try:
            caller = row[caller_idx].strip()
            utt_idx = int(float(row[utt_idx_idx].strip()))
            utt_speaker[conv_id][utt_idx] = caller
        except (ValueError, IndexError):
            continue

    files_checked += 1

print(f"Files checked: {files_checked}, Matched: {matched_files}")
print(f"Conversations with speaker info: {len(utt_speaker)}")

# ========================
# Load splits_final2 and match speaker
# ========================
print("\n=== Loading and matching speaker + DA ===")
speaker_da_counter = {"A": Counter(), "B": Counter()}
matched_utts = 0
unmatched_convs = []

for split in ["train", "valid", "test"]:
    path = f"../data/{split}.jsonl"
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            cid = d["conv_id"]
            tid = d["turn_id"]
            da = d["da"]
            if cid in utt_speaker and tid in utt_speaker[cid]:
                speaker = utt_speaker[cid][tid]
                speaker_da_counter[speaker][da] += 1
                matched_utts += 1
            else:
                if cid not in unmatched_convs:
                    unmatched_convs.append(cid)

print(f"Matched utterances: {matched_utts}")
print(f"Unmatched conversations: {len(set(unmatched_convs))}")
total_a = sum(speaker_da_counter["A"].values())
total_b = sum(speaker_da_counter["B"].values())
print(f"Speaker A: {total_a}, Speaker B: {total_b}")

# ========================
# E2: Speaker DA distribution
# ========================
print("\n=== E2: Speaker DA Distribution ===")
top_das = ["sd", "b", "sv", "+", "%", "aa", "ba", "qy", "x", "ny"]
x = np.arange(len(top_das))
width = 0.35

a_vals = [speaker_da_counter["A"].get(da, 0)/max(total_a,1) for da in top_das]
b_vals = [speaker_da_counter["B"].get(da, 0)/max(total_b,1) for da in top_das]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.bar(x - width/2, a_vals, width, label='Speaker A', color='cornflowerblue')
ax1.bar(x + width/2, b_vals, width, label='Speaker B', color='coral')
ax1.set_xticks(x)
ax1.set_xticklabels(top_das, rotation=45, ha='right')
ax1.set_ylabel('Proportion')
ax1.set_title('E2: DA Distribution by Speaker (Normalized)')
ax1.legend()

top_a = speaker_da_counter["A"].most_common(10)
top_b = speaker_da_counter["B"].most_common(10)
da_labels = [d for d,c in top_a[::-1]]
a_counts = [c for d,c in top_a[::-1]]
b_counts = [speaker_da_counter["B"].get(d, 0) for d in da_labels]
y = np.arange(len(da_labels))

ax2 = axes[1]
ax2.barh(y - width/2, a_counts, width, label='Speaker A', color='cornflowerblue')
ax2.barh(y + width/2, b_counts, width, label='Speaker B', color='coral')
ax2.set_yticks(y)
ax2.set_yticklabels(da_labels)
ax2.set_xlabel('Count')
ax2.set_title('E2: Top-10 DAs by Speaker')
ax2.legend()

plt.tight_layout()
plt.savefig("../figures/E2_speaker_distribution.png", dpi=150)
print("Saved: E2_speaker_distribution.png")

print("\nSpeaker A top5:", speaker_da_counter["A"].most_common(5))
print("Speaker B top5:", speaker_da_counter["B"].most_common(5))

# ========================
# E5: Speaker UMAP - aggregate to speaker level
# ========================
print("\n=== E5: Speaker-level UMAP ===")
VECTORS_FILE = os.path.join(os.path.dirname(__file__), "..", "vectors", "prag_vectors.pt")
if not os.path.exists(VECTORS_FILE):
    raise FileNotFoundError(
        f"prag_vectors.pt not found: {VECTORS_FILE}\n"
        "This file is included in the repository but may need to be downloaded.\n"
        "If missing, run: python scripts/make_contexts_from_swda.py && python scripts/precompute_prag_vectors.py"
    )
vectors = torch.load(VECTORS_FILE, map_location="cpu").numpy()
print(f"prag_vectors shape: {vectors.shape}")

# Aggregate per-utterance pragmatic vectors by speaker
# For E5, we show the DA distribution as a "speaker profile" scatter
# in a 2D space derived from the top DA proportions
speaker_da_2d = np.array([
    [speaker_da_counter["A"].get(da, 0)/max(total_a,1) for da in top_das],
    [speaker_da_counter["B"].get(da, 0)/max(total_b,1) for da in top_das]
])
# Use PCA to reduce to 2D for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
speaker_2d = pca.fit_transform(speaker_da_2d)
print(f"Speaker 2D coords: A={speaker_2d[0]}, B={speaker_2d[1]}")
explained = pca.explained_variance_ratio_
print(f"PCA explained variance: {explained}")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter([speaker_2d[0,0]], [speaker_2d[0,1]], c='cornflowerblue', s=300, label=f'Speaker A (n={total_a})', marker='s', zorder=5)
ax.scatter([speaker_2d[1,0]], [speaker_2d[1,1]], c='coral', s=300, label=f'Speaker B (n={total_b})', marker='s', zorder=5)
ax.annotate(f'Speaker A\n(n={total_a})', (speaker_2d[0,0], speaker_2d[0,1]), textcoords="offset points", xytext=(10,10), fontsize=10)
ax.annotate(f'Speaker B\n(n={total_b})', (speaker_2d[1,0], speaker_2d[1,1]), textcoords="offset points", xytext=(10,-15), fontsize=10)
ax.set_title(f'E5: Speaker Pragmatic Profiles (PCA of DA Distribution)\nVar explained: {explained[0]:.2%} + {explained[1]:.2%} = {sum(explained):.2%}')
ax.legend()
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
plt.tight_layout()
plt.savefig("../figures/E5_speaker_umap.png", dpi=150)
print("Saved: E5_speaker_umap.png")

print("\nAll analysis complete!")