#!/bin/bash
#
# Download and process MapTask v2.1 data for pragmatic-framework
#
# Usage: bash scripts/download_maptask.sh
#
# What this script does:
#   1. Download MapTask v2.1 from official source
#   2. Extract to maptask/ directory
#   3. Extract text+DA from XML files → maptask/maptask_text_da.json
#   4. Create episodes from SwDA test.jsonl → maptask/episodes_T6.jsonl
#   5. Create figures/ output directory
#

set -e

MAPTASK_URL="https://groups.inf.ed.ac.uk/maptask/maptaskv2-1.tar.gz"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Downloading MapTask v2.1 ==="
mkdir -p "$ROOT_DIR/maptask"
if [ -f "$ROOT_DIR/maptask/maptaskv2-1.tar.gz" ]; then
    echo "Archive already exists, skipping download."
else
    curl -L -o "$ROOT_DIR/maptask/maptaskv2-1.tar.gz" "$MAPTASK_URL"
fi

echo "=== Extracting MapTask v2.1 ==="
tar -xzf "$ROOT_DIR/maptask/maptaskv2-1.tar.gz" -C "$ROOT_DIR/maptask/"
echo "Extracted to $ROOT_DIR/maptask/"

echo "=== Extracting text + DA from MapTask XML ==="
python3 - << 'PYTHON_EOF'
import os
import json
import xml.etree.ElementTree as ET
from collections import Counter

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = os.path.join(root_dir, "maptask", "maptaskv2-1", "Data")
moves_dir = os.path.join(base, "moves")
tu_dir = os.path.join(base, "timed-units")
out_file = os.path.join(root_dir, "maptask", "maptask_text_da.json")

NS = "http://nite.sourceforge.net/"

def get_text_from_timed_units(tu_file, tu_start, tu_end):
    try:
        tree = ET.parse(tu_file)
        root = tree.getroot()
    except:
        return ""
    try:
        start_num = int(tu_start.split(".")[-1])
        end_num = int(tu_end.split(".")[-1])
    except:
        return ""
    texts = []
    for tu in root.iter("tu"):
        tu_id = tu.get("id", "")
        try:
            tu_num = int(tu_id.split(".")[-1])
        except:
            continue
        if start_num <= tu_num <= end_num:
            text = (tu.text or "").strip()
            if text:
                texts.append(text)
    return " ".join(texts)

def extract_utterances(moves_file, tu_file):
    tree = ET.parse(moves_file)
    root = tree.getroot()
    utterances = []
    for move in root.iter("move"):
        da_label = move.get("label", "")
        if not da_label:
            continue
        for child in move.iter(f"{{{NS}}}child"):
            href = child.get("href", "")
            if "timed-units" not in href or "#id(" not in href:
                continue
            ids_str = href.split("#id(")[1].rstrip(")")
            if ".." in ids_str:
                parts = ids_str.split("..id(")
                tu_start = parts[0]
                tu_end = parts[1]
            else:
                tu_start = ids_str
                tu_end = ids_str
            text = get_text_from_timed_units(tu_file, tu_start, tu_end)
            if text and len(text) > 1:
                utterances.append({"text": text, "da": da_label})
            break
    return utterances

moves_files = sorted([f for f in os.listdir(moves_dir) if f.endswith(".moves.xml")])
all_utterances = []
for mf in moves_files:
    parts = mf.replace(".moves.xml", "").split(".")
    if len(parts) < 2:
        continue
    conv_id = parts[0]
    speaker_code = parts[1]
    moves_file = os.path.join(moves_dir, mf)
    tu_file = os.path.join(tu_dir, f"{conv_id}.{speaker_code}.timed-units.xml")
    if not os.path.exists(tu_file):
        continue
    try:
        utterances = extract_utterances(moves_file, tu_file)
        for u in utterances:
            u["conv_id"] = conv_id
            u["speaker"] = speaker_code
        all_utterances.extend(utterances)
    except Exception as e:
        print(f"Error on {mf}: {e}")

print(f"Total utterances: {len(all_utterances)}")
da_counter = Counter(u["da"] for u in all_utterances)
print(f"DA types: {len(da_counter)}")
for da, cnt in da_counter.most_common():
    print(f"  {da}: {cnt}")

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(all_utterances, f, ensure_ascii=False)
print(f"Saved: {out_file}")
PYTHON_EOF

echo "=== Building SwDA episodes ==="
python3 - << 'PYTHON_EOF'
import os
import json
from collections import defaultdict

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = os.path.join(root_dir, "data", "test.jsonl")
out = os.path.join(root_dir, "maptask", "episodes_T6.jsonl")
T = 6
MAX_EP = 200

by_conv = defaultdict(list)
with open(src, "r") as f:
    for line in f:
        r = json.loads(line)
        by_conv[r["conv_id"]].append(r)

episodes = []
for cid, rows in by_conv.items():
    rows = sorted(rows, key=lambda x: x["turn_id"])
    texts = [x["text"].strip() for x in rows if x.get("text", "").strip()]
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

with open(out, "w") as w:
    for e in episodes:
        w.write(json.dumps(e, ensure_ascii=False) + "\n")
print(f"Episodes: {len(episodes)} -> {out}")
PYTHON_EOF

echo "=== Creating figures directory ==="
mkdir -p "$ROOT_DIR/figures"
echo "Done. figures/ directory ready."
echo ""
echo "MapTask data is ready at: $ROOT_DIR/maptask/"
echo "You can now run the analysis scripts."
