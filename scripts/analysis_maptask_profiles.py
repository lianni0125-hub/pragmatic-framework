#!/usr/bin/env python3
"""Analyze speaker and topic profiles from MapTask corpus."""
import xml.etree.ElementTree as ET
import os, re
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
MAPTASK_CORPUS_XML = os.path.join(ROOT_DIR, "maptask", "maptaskv2-1", "Data", "corpus-resources", "maptask-corpus.xml")
if not os.path.exists(MAPTASK_CORPUS_XML):
    raise FileNotFoundError(
        f"MapTask data not found: {MAPTASK_CORPUS_XML}\n"
        "Run: bash scripts/download_maptask.sh"
    )

BASE = os.path.join(ROOT_DIR, "maptask", "maptaskv2-1", "Data")
NS = "http://nite.sourceforge.net/"

# ========================
# Step 1: Parse corpus metadata (conv -> map, speakers)
# Note: conv elements are in null namespace, pointer elements are in NS namespace
# ========================
print("=== Step 1: Parsing corpus metadata ===")
corpus_tree = ET.parse(f"{BASE}/corpus-resources/maptask-corpus.xml")
corpus_root = corpus_tree.getroot()

conv_map = {}  # conv_id -> map_id
conv_speakers = {}  # conv_id -> {g: speaker_id, f: speaker_id}

for conv in corpus_root.iter("conv"):
    conv_id = conv.get("id")
    map_id = conv.get("map")
    conv_map[conv_id] = map_id
    agents = {}
    for pointer in conv.iter(f"{{{NS}}}pointer"):
        role = pointer.get("role")
        href = pointer.get("href")
        speaker_id = href.split("id(")[1].rstrip(")")
        agents[role] = speaker_id
    conv_speakers[conv_id] = agents

# Participants: speaker elements in null namespace
part_tree = ET.parse(f"{BASE}/corpus-resources/maptask-participants.xml")
part_root = part_tree.getroot()
speaker_info = {}  # speaker_id -> {name, gender}
for sp in part_root.iter("speaker"):
    sid = sp.get("id")
    speaker_info[sid] = {"name": sp.get("name"), "gender": sp.get("gender")}

print(f"Conversations: {len(conv_map)}, Participants: {len(speaker_info)}")
print(f"Sample: conv_map={list(conv_map.items())[:2]}, conv_speakers={dict(list(conv_speakers.items())[:2])}")

# ========================
# Step 2: Parse all moves XML files directly
# ========================
print("\n=== Step 2: Parsing moves XML files ===")
moves_dir = f"{BASE}/moves"
moves_files = [f for f in os.listdir(moves_dir) if f.endswith(".moves.xml")]
print(f"Moves files: {len(moves_files)}")

utterances = []  # {conv_id, speaker, speaker_code, da, map_id}

for mf in moves_files:
    # Parse filename: q1ec1.g.moves.xml -> conv_id=q1ec1, speaker_code=g
    parts = mf.replace(".moves.xml", "").split(".")
    if len(parts) < 2:
        continue
    conv_id = parts[0]  # e.g. q1ec1
    speaker_code = parts[1]  # g or f

    if conv_id not in conv_map:
        continue
    map_id = conv_map[conv_id]

    # Get speaker real name
    sp_id = conv_speakers.get(conv_id, {}).get(speaker_code, "")
    speaker_name = speaker_info.get(sp_id, {}).get("name", speaker_code)
    gender = speaker_info.get(sp_id, {}).get("gender", "unknown")

    moves_path = f"{moves_dir}/{mf}"
    try:
        moves_tree = ET.parse(moves_path)
        moves_root = moves_tree.getroot()
    except:
        continue

    for move in moves_root.iter("move"):
        da_label = move.get("label", "")
        if da_label:
            utterances.append({
                "conv_id": conv_id,
                "speaker": speaker_name,
                "speaker_code": speaker_code,
                "gender": gender,
                "da": da_label,
                "map_id": map_id,
            })

print(f"Total utterances: {len(utterances)}")
if not utterances:
    print("ERROR: No utterances extracted!")
else:
    da_counts = Counter(u["da"] for u in utterances)
    print(f"DA types: {len(da_counts)}")
    print(f"Top DAs: {da_counts.most_common(10)}")
    print(f"Top speakers: {Counter(u['speaker'] for u in utterances).most_common(5)}")

    # ========================
    # E4: Cross-topic DA distribution
    # ========================
    print("\n=== E4: Cross-topic DA Distribution ===")
    topic_da = defaultdict(Counter)
    for u in utterances:
        topic_da[u["map_id"]][u["da"]] += 1

    top_das = [d for d, c in da_counts.most_common(8)]
    topic_ids = sorted(set(u["map_id"] for u in utterances))
    print(f"Topics (maps): {topic_ids}")

    if len(topic_ids) > 0 and len(top_das) > 0:
        # Heatmap
        mat = np.zeros((len(topic_ids), len(top_das)))
        for i, tid in enumerate(topic_ids):
            total = sum(topic_da[tid].values())
            for j, da in enumerate(top_das):
                mat[i, j] = topic_da[tid].get(da, 0) / max(total, 1)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(mat, xticklabels=top_das, yticklabels=topic_ids,
                    cmap="YlOrRd", ax=ax, vmin=0, vmax=0.6)
        ax.set_xlabel("Dialogue Act")
        ax.set_ylabel("Topic (Map ID)")
        ax.set_title("E4: Cross-Topic DA Distribution (MapTask)")
        plt.tight_layout()
        plt.savefig("../figures/E4_topic_da_distribution.png", dpi=150)
        print("Saved: E4_topic_da_distribution.png")

        # Topic detail charts
        n_topics = min(8, len(topic_ids))
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(n_topics):
            tid = topic_ids[i]
            ax = axes[i//4, i%4]
            total = sum(topic_da[tid].values())
            top5 = topic_da[tid].most_common(5)
            das = [d for d,c in top5]
            cnts = [c/max(total,1) for d,c in top5]
            ax.barh(das[::-1], cnts[::-1], color="steelblue")
            ax.set_title(f"Map {tid} (n={total})")
            ax.set_xlim(0, 1)
        plt.suptitle("E4: Top-5 DAs per Topic (MapTask)", y=1.01)
        plt.tight_layout()
        plt.savefig("../figures/E4_topic_da_detail.png", dpi=150)
        print("Saved: E4_topic_da_detail.png")

    # ========================
    # E6: Real speaker profiling
    # ========================
    print("\n=== E6: Real Speaker Profiling ===")
    speaker_da = defaultdict(Counter)
    speaker_gender_da = defaultdict(Counter)
    for u in utterances:
        speaker_da[u["speaker"]][u["da"]] += 1
        speaker_gender_da[u["gender"]][u["da"]] += 1

    speaker_total = {sp: sum(c.values()) for sp, c in speaker_da.items()}
    print(f"Unique speakers: {len(speaker_da)}")
    print(f"Top speakers: {sorted(speaker_total.items(), key=lambda x: -x[1])[:5]}")

    # E6a: Speaker comparison (top speakers)
    top_speakers = [sp for sp, t in sorted(speaker_total.items(), key=lambda x: -x[1])[:12]]
    if len(top_speakers) > 1:
        top_speakers_da = [d for d, c in da_counts.most_common(6)]
        x = np.arange(len(top_speakers_da))

        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_speakers)))
        w = 0.8 / len(top_speakers)
        for i, sp in enumerate(top_speakers):
            total = sum(speaker_da[sp].values())
            vals = [speaker_da[sp].get(da, 0)/max(total,1) for da in top_speakers_da]
            ax.bar(x + i*w - (len(top_speakers)-1)*w/2, vals, w, label=sp, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(top_speakers_da, rotation=30, ha="right")
        ax.set_ylabel("Proportion")
        ax.set_title("E6: DA Distribution by Individual Speaker (MapTask)")
        ax.legend(fontsize=6, ncol=min(4, len(top_speakers)), loc="upper right")
        plt.tight_layout()
        plt.savefig("../figures/E6_speaker_profiling.png", dpi=150)
        print("Saved: E6_speaker_profiling.png")

    # E6b: Gender comparison
    gender_total = {g: sum(speaker_gender_da[g].values()) for g in speaker_gender_da}
    print(f"\nGender distribution: {gender_total}")
    for g, total in gender_total.items():
        top5 = speaker_gender_da[g].most_common(5)
        print(f"  {g}: {total} utts, top DAs: {top5}")

    top_das_gender = [d for d, c in da_counts.most_common(6)]
    x = np.arange(len(top_das_gender))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    color_map = {"m": "cornflowerblue", "f": "coral", "unknown": "gray"}
    label_map = {"m": "Male", "f": "Female", "unknown": "Unknown"}
    for i, g in enumerate(["m", "f"]):
        total = gender_total.get(g, 0)
        if total == 0:
            continue
        vals = [speaker_gender_da[g].get(da, 0)/max(total,1) for da in top_das_gender]
        ax.bar(x + (i-0.5)*width, vals, width, label=label_map.get(g,g), color=color_map.get(g,"gray"))
    ax.set_xticks(x)
    ax.set_xticklabels(top_das_gender, rotation=30, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title("E6: DA Distribution by Speaker Gender (MapTask)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/E6_gender_comparison.png", dpi=150)
    print("Saved: E6_gender_comparison.png")

    # ========================
    # Speaker UMAP (E5 equivalent) - aggregate by speaker
    # ========================
    print("\n=== E5: Speaker-level UMAP ===")
    from sklearn.decomposition import PCA

    top_das_for_vec = [d for d, c in da_counts.most_common(8)]
    unique_speakers = list(speaker_da.keys())
    speaker_vec = np.zeros((len(unique_speakers), len(top_das_for_vec)))
    for i, sp in enumerate(unique_speakers):
        total = sum(speaker_da[sp].values())
        for j, da in enumerate(top_das_for_vec):
            speaker_vec[i, j] = speaker_da[sp].get(da, 0) / max(total, 1)

    n_components = min(2, len(unique_speakers)-1, len(top_das_for_vec)-1)
    pca = PCA(n_components=max(1, n_components))
    speaker_2d = pca.fit_transform(speaker_vec)

    fig, ax = plt.subplots(figsize=(8, 6))
    sp_genders = {}  # speaker -> gender
    for u in utterances:
        sp_genders[u["speaker"]] = u["gender"]
    for i, sp in enumerate(unique_speakers):
        g = sp_genders.get(sp, "unknown")
        color = "cornflowerblue" if g == "m" else ("coral" if g == "f" else "gray")
        ax.scatter(speaker_2d[i, 0], speaker_2d[i, 1], c=color, s=80, alpha=0.7)
        ax.annotate(sp, (speaker_2d[i, 0]+0.01, speaker_2d[i, 1]+0.01), fontsize=5)
    ax.set_title("E5: Speaker Pragmatic Profiles (PCA of DA Distribution)")
    ax.legend(handles=[plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='cornflowerblue', label='Male', markersize=8),
                       plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='coral', label='Female', markersize=8)])
    plt.tight_layout()
    plt.savefig("../figures/E5_speaker_umap.png", dpi=150)
    print("Saved: E5_speaker_umap.png")

    print("\nAll MapTask analysis complete!")