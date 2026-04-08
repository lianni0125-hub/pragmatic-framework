#!/usr/bin/env python3
"""Analyze speaker profiles from MapTask conversation episodes."""
import json, numpy as np, re, os
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch

EPISODES_FILE = os.path.join(os.path.dirname(__file__), "..", "maptask", "episodes_T6.jsonl")
if not os.path.exists(EPISODES_FILE):
    raise FileNotFoundError(
        f"Episodes file not found: {EPISODES_FILE}\n"
        "Run: bash scripts/download_maptask.sh"
    )

# ========================
# Speaker info from episodes_T6
# ========================
print("=== Loading speaker-utterance data ===")
# Parse episodes_T6 to get speaker per turn
episode_speaker_turns = {}  # conv_id -> {turn_id -> speaker}
conv_ids_in_episodes = set()

with open(EPISODES_FILE) as f:
    for line in f:
        ep = json.loads(line)
        cid = ep["conv_id"]
        turn0 = ep["turn0"]
        conv_ids_in_episodes.add(cid)
        episode_speaker_turns[cid] = {}

        for i, utt_text in enumerate(ep["turns"]):
            turn_id = turn0 + i
            # Extract speaker from text like "B.1 utt1:" or "A.2 utt1:"
            m = re.match(r'^([AB])\.\d+\s+utt\d+:', utt_text.strip())
            if m:
                speaker = m.group(1)
            elif utt_text.startswith("I "):
                # First turn often just text, assume alternating A/B based on position
                speaker = "A" if i % 2 == 0 else "B"
            else:
                speaker = "B" if i % 2 == 1 else "A"
            episode_speaker_turns[cid][turn_id] = speaker

print(f"Conversations in episodes: {len(conv_ids_in_episodes)}")
# Sample check
sample_cid = list(conv_ids_in_episodes)[0]
print(f"Sample conv {sample_cid} turns:")
for tid, sp in list(episode_speaker_turns[sample_cid].items())[:5]:
    print(f"  turn {tid}: speaker {sp}")

# ========================
# Load all utterances with DA from splits_final2 (test set only for speed)
# ========================
print("\n=== Loading DA labels from test set ===")
test_utterances = []
conv_turn_text = defaultdict(dict)

for split in ["test"]:
    path = f"../data/{split}.jsonl"
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            cid = d["conv_id"]
            tid = d["turn_id"]
            text = d["text"]
            da = d["da"]
            conv_turn_text[cid][tid] = {"text": text, "da": da}

print(f"Conversations in test: {len(conv_turn_text)}")

# ========================
# Match: use conversations that appear in both
# ========================
matched_convs = conv_ids_in_episodes & set(conv_turn_text.keys())
print(f"Matched conversations: {len(matched_convs)}")

speaker_da_counter = {"A": Counter(), "B": Counter()}
for cid in matched_convs:
    ep_turns = episode_speaker_turns[cid]
    sp2da = conv_turn_text[cid]
    for tid, speaker in ep_turns.items():
        if tid in sp2da:
            speaker_da_counter[speaker][sp2da[tid]["da"]] += 1

total_a = sum(speaker_da_counter["A"].values())
total_b = sum(speaker_da_counter["B"].values())
print(f"\nSpeaker A utterances: {total_a}")
print(f"Speaker B utterances: {total_b}")

# ========================
# E2: Speaker DA distribution comparison
# ========================
print("\n=== E2: Speaker DA Distribution ===")
top_das = ["sd", "b", "sv", "+", "%", "aa", "ba", "qy", "x", "ny"]
x = np.arange(len(top_das))
width = 0.35

a_vals = [speaker_da_counter["A"].get(da, 0)/max(total_a,1) for da in top_das]
b_vals = [speaker_da_counter["B"].get(da, 0)/max(total_b,1) for da in top_das]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart: normalized comparison
ax1 = axes[0]
ax1.bar(x - width/2, a_vals, width, label='Speaker A', color='cornflowerblue')
ax1.bar(x + width/2, b_vals, width, label='Speaker B', color='coral')
ax1.set_xticks(x)
ax1.set_xticklabels(top_das, rotation=45, ha='right')
ax1.set_ylabel('Proportion')
ax1.set_title('E2: DA Distribution by Speaker (Normalized)')
ax1.legend()

# Speaker A top DAS
ax2 = axes[1]
top_a = speaker_da_counter["A"].most_common(10)
ax2.barh([d for d,c in top_a[::-1]], [c for d,c in top_a[::-1]], color='cornflowerblue')
ax2.set_xlabel('Count')
ax2.set_title('E2: Speaker A Top-10 DAs')

plt.tight_layout()
plt.savefig("../figures/E2_speaker_distribution.png", dpi=150)
print("Saved: E2_speaker_distribution.png")

# ========================
# E5: Speaker UMAP from prag_vectors.pt
# ========================
print("\n=== E5: Speaker UMAP ===")
# Load prag_vectors.pt (100 x 768)
VECTORS_FILE = os.path.join(os.path.dirname(__file__), "..", "vectors", "prag_vectors.pt")
if not os.path.exists(VECTORS_FILE):
    raise FileNotFoundError(
        f"prag_vectors.pt not found: {VECTORS_FILE}\n"
        "This file is included in the repository but may need to be downloaded.\n"
        "If missing, run: python scripts/make_contexts_from_swda.py && python scripts/precompute_prag_vectors.py"
    )
vectors = torch.load(VECTORS_FILE, map_location="cpu").numpy()
print(f"prag_vectors shape: {vectors.shape}")

# We have 100 vectors but need speaker labels
# The vectors were computed from contexts.txt (100 contexts)
# Each context corresponds to an episode in episodes_T6
# episodes_T6 has 200 lines -> 100 vectors (2 turns per vector?)
# Let's pair vectors with speaker info from first 100 episodes

# Load first 100 episodes and their speakers
speaker_labels = []
with open(EPISODES_FILE) as f:
    for idx, line in enumerate(f):
        if idx >= 100:
            break
        ep = json.loads(line)
        cid = ep["conv_id"]
        # Get first turn's speaker as episode-level speaker
        first_turn_text = ep["turns"][0]
        m = re.match(r'^([AB])\.\d+\s+utt\d+:', first_turn_text.strip())
        if m:
            sp = m.group(1)
        else:
            sp = "A"
        speaker_labels.append(sp)

print(f"Vector count: {vectors.shape[0]}, Speaker labels: {len(speaker_labels)}")
a_count = speaker_labels.count("A")
b_count = speaker_labels.count("B")
print(f"Speaker A: {a_count}, Speaker B: {b_count}")

# UMAP (use t-SNE since sklearnUMAP may not be installed)
from sklearn.manifold import TSNE
vecs_2d = TSNE(n_components=2, perplexity=min(30, len(vectors)-1), random_state=42).fit_transform(vectors)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["cornflowerblue" if sp=="A" else "coral" for sp in speaker_labels]
ax.scatter(vecs_2d[:,0], vecs_2d[:,1], c=colors, alpha=0.6, s=30)
ax.set_title("E5: Pragmatic Representation Space Colored by Speaker (t-SNE)")
ax.legend(handles=[plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='cornflowerblue', label='Speaker A', markersize=8),
                   plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='coral', label='Speaker B', markersize=8)])
plt.tight_layout()
plt.savefig("../figures/E5_speaker_umap.png", dpi=150)
print("Saved: E5_speaker_umap.png")

print("\nE2 and E5 complete!")