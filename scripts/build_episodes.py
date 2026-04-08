#!/usr/bin/env python3
"""Build conversation episodes from test.jsonl for speaker analysis."""
import json
import argparse
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser(description="Build conversation episodes from test data")
parser.add_argument("--src", default=None,
    help="Input JSONL file (default: <root>/data/test.jsonl)")
parser.add_argument("--out", default=None,
    help="Output JSONL file (default: <root>/maptask/episodes_T6.jsonl)")
parser.add_argument("--turns", "-t", type=int, default=6,
    help="Turns per episode (default: 6)")
parser.add_argument("--max-ep", type=int, default=200,
    help="Maximum number of episodes to generate (default: 200)")
args = parser.parse_args()

SRC = args.src or os.path.join(ROOT_DIR, "data", "test.jsonl")
OUT = args.out or os.path.join(ROOT_DIR, "maptask", "episodes_T6.jsonl")
T = args.turns
MAX_EP = args.max_ep

by_conv = defaultdict(list)
with open(SRC, "r") as f:
    for line in f:
        r = json.loads(line)
        by_conv[r["conv_id"]].append(r)

episodes = []
for cid, rows in by_conv.items():
    rows = sorted(rows, key=lambda x: x["turn_id"])
    # Extract text from each turn
    texts = [x["text"].strip() for x in rows if x.get("text","").strip()]
    if len(texts) < T:
        continue
    for i in range(0, len(texts) - T + 1, T):
        ep = texts[i:i+T]
        if len(ep) == T:
            episodes.append({"conv_id": cid, "turn0": i, "turns": ep})
            if len(episodes) >= MAX_EP:
                break
    if len(episodes) >= MAX_EP:
        break

with open(OUT, "w") as w:
    for e in episodes:
        w.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"episodes: {len(episodes)} -> {OUT}")
