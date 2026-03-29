import json, numpy as np
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# E1: DA distribution
print("=== E1: Global DA Distribution ===")
da_counter = Counter()
for split in ["train", "valid", "test"]:
    path = f"../data/{split}.jsonl"  # TODO: set to your data directory
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            da_counter[d["da"]] += 1

total = sum(da_counter.values())
top20 = da_counter.most_common(20)
print(f"Total: {total}")
for da, cnt in top20:
    print(f"  {da}: {cnt} ({100*cnt/total:.1f}%)")

fig, ax = plt.subplots(figsize=(12, 6))
das = [x[0] for x in top20]
cnts = [x[1] for x in top20]
ax.bar(range(len(das)), cnts, color="steelblue")
ax.set_xticks(range(len(das)))
ax.set_xticklabels(das, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Count")
ax.set_title("E1: Top-20 Dialogue Act Distribution (Switchboard)")
for i, (da, cnt) in enumerate(top20):
    ax.text(i, cnt + total*0.01, f"{100*cnt/total:.1f}%", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig("E1_da_distribution.png", dpi=150)
print("Saved: E1_da_distribution.png")

# E3: Transition matrix
print()
print("=== E3: Pragmatic Transition Matrix ===")
transitions = defaultdict(Counter)
conv_seq = defaultdict(list)
for split in ["train", "valid", "test"]:
    path = f"../data/{split}.jsonl"  # TODO: set to your data directory
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            conv_seq[d["conv_id"]].append((d["turn_id"], d["da"]))

for cid, turns in conv_seq.items():
    turns_sorted = sorted(turns, key=lambda x: x[0])
    for i in range(len(turns_sorted)-1):
        if turns_sorted[i+1][0] == turns_sorted[i][0] + 1:
            transitions[turns_sorted[i][1]][turns_sorted[i+1][1]] += 1

top15 = [x[0] for x in top20[:15]]
n = len(top15)
mat = np.zeros((n, n))
for i, fda in enumerate(top15):
    for j, tda in enumerate(top15):
        mat[i,j] = transitions[fda].get(tda, 0)

row_sums = mat.sum(axis=1, keepdims=True)
row_sums[row_sums==0] = 1
mat_norm = mat / row_sums

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(mat_norm, xticklabels=top15, yticklabels=top15, cmap="Blues", ax=ax, vmin=0, vmax=0.5)
ax.set_xlabel("To DA")
ax.set_ylabel("From DA")
ax.set_title("E3: DA Transition Probability Matrix (Top-15 DAs)")
plt.tight_layout()
plt.savefig("E3_transition_matrix.png", dpi=150)
print("Saved: E3_transition_matrix.png")

all_t = []
for i, fda in enumerate(top15):
    for j, tda in enumerate(top15):
        if mat_norm[i,j] > 0.05:
            all_t.append((fda, tda, mat_norm[i,j]))
all_t.sort(key=lambda x: -x[2])
print("Top transitions:")
for t in all_t[:15]:
    print(f"  {t[0]} -> {t[1]}: {t[2]:.3f}")
print("Done!")