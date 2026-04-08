#!/usr/bin/env python3
"""Discover new pragmatic tags via clustering semantic anomaly pairs."""
import os, json, random, argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser(description="Discover new pragmatic tags via clustering")
parser.add_argument("--test-jsonl", default=None,
    help="Path to test JSONL (default: <root>/data/test.jsonl)")
parser.add_argument("--plugin-dir", default=None,
    help="Path to plugin directory (default: <root>/plugin)")
parser.add_argument("--out-dir", default=None,
    help="Output directory (default: <root>/data/prag_discovery)")
parser.add_argument("--n-samples", type=int, default=4000,
    help="Number of test samples to use (default: 4000)")
args = parser.parse_args()

TEST_JSONL = args.test_jsonl or os.path.join(ROOT_DIR, "data", "test.jsonl")
PRAG_DIR = args.plugin_dir or os.path.join(ROOT_DIR, "plugin")
OUT_DIR = args.out_dir or os.path.join(ROOT_DIR, "data", "prag_discovery")

BASE_NAME  = “distilbert-base-uncased”

MAX_LEN = 64
DEVICE = “cuda” if torch.cuda.is_available() else “cpu”

# A: Sampling pairs & semantic threshold
N_SAMPLES = args.n_samples   # How many test samples to use for discovery
N_PAIRS   = 20000         # Number of random pairs (larger = more thorough but slower)
SEM_THR   = 0.95          # Semantic similarity threshold: high similarity
MAX_ANOM  = 2500          # Maximum anomaly pairs to keep (prevents excessive output)

# B: Clustering parameters
MIN_CLUSTER_SIZE = 25
MIN_SAMPLES      = 10

OUT_DIR = “../data/prag_discovery”
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# =======================
# Load label map
# =======================
with open(os.path.join(PRAG_DIR, "label_map.json"), "r") as f:
    lm = json.load(f)
lab2id = lm["lab2id"]
id2lab = {int(k): v for k, v in lm["id2lab"].items()} if isinstance(list(lm["id2lab"].keys())[0], str) else lm["id2lab"]

# =======================
# Define Prag Encoder (must match your training)
# =======================
class PragPlugin(nn.Module):
    def __init__(self, base_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_name)
        hid = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hid, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        prag_vec = out.last_hidden_state[:, 0, :]
        logits = self.classifier(prag_vec)
        return prag_vec, logits

def load_test_rows(path, n_samples):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    # Filter out empty text
    rows = [r for r in rows if isinstance(r.get("text",""), str) and r["text"].strip()]
    if len(rows) > n_samples:
        rows = random.sample(rows, n_samples)
    return rows

def batch_encode(tok, texts):
    enc = tok(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)

@torch.no_grad()
def get_vectors(prag_model, base_model, tok, rows, bs=64):
    texts = [r["text"] for r in rows]
    das   = [r["da"] for r in rows]
    ids   = list(range(len(rows)))

    prag_vecs = []
    sem_vecs  = []
    for i in tqdm(range(0, len(texts), bs), desc="Embedding"):
        chunk = texts[i:i+bs]
        input_ids, attn = batch_encode(tok, chunk)

        # pragmatic vec from prag encoder (same as training: CLS)
        pv, _ = prag_model(input_ids=input_ids, attention_mask=attn)
        prag_vecs.append(pv.detach().cpu().numpy())

        # semantic vec from base encoder CLS
        out = base_model(input_ids=input_ids, attention_mask=attn)
        sv = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
        sem_vecs.append(sv)

    prag_vecs = np.concatenate(prag_vecs, axis=0)
    sem_vecs  = np.concatenate(sem_vecs, axis=0)

    return ids, texts, das, prag_vecs, sem_vecs

def semantic_anomaly_pairs(das, sem_vecs, n_pairs, sem_thr, max_keep):
    n = len(das)
    keep = []
    # Normalize vectors for cosine similarity
    sem = sem_vecs / (np.linalg.norm(sem_vecs, axis=1, keepdims=True) + 1e-9)

    for _ in tqdm(range(n_pairs), desc=”Sampling pairs”):
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j:
            continue
        sim = float(np.dot(sem[i], sem[j]))
        if sim >= sem_thr and das[i] != das[j]:
            keep.append((i, j, sim))
            if len(keep) >= max_keep:
                break
    return keep

def draft_descriptor(cluster_texts, cluster_das):
    # Simple rule-based descriptive tags (no invented notation)
    top_da = Counter(cluster_das).most_common(3)
    da_str = “, “.join([f”{d}({c})” for d, c in top_da])

    # Simple surface features (discourse markers, question marks, negations)
    markers = {
        "question_like": sum("?" in t for t in cluster_texts),
        "negation": sum(("n't" in t.lower()) or (" not " in t.lower()) for t in cluster_texts),
        "short_utter": sum(len(t.split()) <= 5 for t in cluster_texts),
        "discourse_markers": sum(t.lower().strip().startswith(("well", "so", "but", "and", "yeah", "no", "um", "uh")) for t in cluster_texts),
    }
    # Select top two most prominent features
    top2 = sorted(markers.items(), key=lambda x: x[1], reverse=True)[:2]
    feat = ", ".join([f"{k}={v}" for k, v in top2])

    return f"DA-mix: {da_str} | surface: {feat}"

def main():
    print("DEVICE:", DEVICE)

    # load tokenizer
    tok = AutoTokenizer.from_pretrained(PRAG_DIR)

    # load prag model weights
    prag_model = PragPlugin(BASE_NAME, num_labels=len(lab2id))
    # Load from model.safetensors
    from safetensors.torch import load_file
    st = load_file(os.path.join(PRAG_DIR, "model.safetensors"))
    missing, unexpected = prag_model.load_state_dict(st, strict=False)
    print("Loaded safetensors. missing:", len(missing), "unexpected:", len(unexpected))
    prag_model.to(DEVICE).eval()

    # base semantic encoder
    base_model = AutoModel.from_pretrained(BASE_NAME).to(DEVICE).eval()

    # load data
    rows = load_test_rows(TEST_JSONL, N_SAMPLES)
    print("Loaded rows:", len(rows))

    ids, texts, das, prag_vecs, sem_vecs = get_vectors(prag_model, base_model, tok, rows, bs=64)

    # =======================
    # A) Find "pragmatic anomaly" region:
    # semantic high-sim but DA different
    # =======================
    pairs = semantic_anomaly_pairs(das, sem_vecs, N_PAIRS, SEM_THR, MAX_ANOM)
    print("Kept anomaly pairs:", len(pairs))

    # Build anomaly set (unique utterance indices involved)
    anom_idx = sorted(set([i for i,_,_ in pairs] + [j for _,j,_ in pairs]))
    print("Unique anomaly utterances:", len(anom_idx))

    # Save A result
    a_out = os.path.join(OUT_DIR, "A_anomaly_pairs.jsonl")
    with open(a_out, "w") as f:
        for i, j, sim in pairs:
            f.write(json.dumps({
                "i": i, "j": j, "sem_sim": sim,
                "text_i": texts[i], "da_i": das[i],
                "text_j": texts[j], "da_j": das[j],
            }, ensure_ascii=False) + "\n")
    print("Saved A:", a_out)

    # =======================
    # B) Cluster pragmatic vectors of anomaly utterances
    # =======================
    X = prag_vecs[anom_idx]
    # Normalize for cosine-like geometry
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(X)
    # -1 means noise (unclustered)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Clusters:", n_clusters, "| noise:", int((labels==-1).sum()))

    # =======================
    # C) Summarise clusters into "descriptive tags"
    # =======================
    # map back to original utterances
    cluster_map = defaultdict(list)  # cl -> [orig_index]
    for local_k, cl in enumerate(labels):
        if cl == -1:
            continue
        cluster_map[int(cl)].append(anom_idx[local_k])

    report = []
    for cl, idxs in sorted(cluster_map.items(), key=lambda x: -len(x[1])):
        cl_texts = [texts[i] for i in idxs]
        cl_das   = [das[i] for i in idxs]
        da_cnt   = Counter(cl_das)
        top_da   = da_cnt.most_common(10)

        # prototypes: pick closest to centroid in prag space
        V = prag_vecs[idxs]
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        centroid = Vn.mean(axis=0, keepdims=True)
        sims = cosine_similarity(Vn, centroid).reshape(-1)
        proto_order = np.argsort(-sims)[:5]
        prototypes = [{
            "text": cl_texts[p],
            "da": cl_das[p],
            "centroid_sim": float(sims[p])
        } for p in proto_order]

        desc = draft_descriptor(cl_texts, cl_das)

        report.append({
            "cluster_id": cl,
            "size": len(idxs),
            "da_top": top_da,
            "descriptor": desc,
            "prototypes": prototypes,
        })

    c_out = os.path.join(OUT_DIR, "C_cluster_report.json")
    with open(c_out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Saved C report:", c_out)

    # quick table for viewing
    tab = []
    for r in report[:15]:
        tab.append({
            "cluster": r["cluster_id"],
            "size": r["size"],
            "top_DA": ", ".join([f"{d}:{c}" for d,c in r["da_top"][:3]]),
            "descriptor": r["descriptor"]
        })
    df = pd.DataFrame(tab)
    t_out = os.path.join(OUT_DIR, "C_cluster_table.csv")
    df.to_csv(t_out, index=False)
    print("Saved table:", t_out)

if __name__ == "__main__":
    main()
