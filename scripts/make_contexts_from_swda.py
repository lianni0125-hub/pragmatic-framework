#!/usr/bin/env python3
"""Build context file for pragmatic vector precomputation."""
import json, random, os, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser(description="Build context file for vector precomputation")
parser.add_argument("--src", default=None, help="Input JSONL (default: <root>/data/test.jsonl)")
parser.add_argument("--out", default=None, help="Output contexts file (default: <root>/data/contexts.txt)")
parser.add_argument("-n", type=int, default=100, help="Number of contexts (default: 100)")
args = parser.parse_args()

SRC = args.src or os.path.join(ROOT_DIR, "data", "test.jsonl")
OUT = args.out or os.path.join(ROOT_DIR, "data", "contexts.txt")
N = args.n
SEED = 42
MAX_CHARS = 180

random.seed(SEED)

rows=[]
with open(SRC,"r") as f:
    for line in f:
        line=line.strip()
        if not line: 
            continue
        r=json.loads(line)
        t=r.get("text","").strip()
        if not t:
            continue
        # Filter out utterances that are too short
        if len(t) < 8:
            continue
        # Remove SwDA annotation markers
        t = t.replace("{F ", "").replace("{C ", "").replace("}", "")
        t = " ".join(t.split())
        if len(t) > MAX_CHARS:
            t = t[:MAX_CHARS].rsplit(" ",1)[0]
        rows.append(t)

random.shuffle(rows)
rows = rows[:N]

with open(OUT,"w") as w:
    for t in rows:
        w.write(f"A: {t}\nB:\n\n")

print("Wrote contexts:", len(rows), "->", OUT)
