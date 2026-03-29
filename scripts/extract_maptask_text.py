"""
Extract text + gold DA from MapTask XML files (corrected).
timed-units XML contains text directly as element content.
Strategy: moves XML -> timed-unit range -> timed-units XML -> text
"""
import os
import json
import xml.etree.ElementTree as ET
from collections import Counter

NS = "http://nite.sourceforge.net/"

def get_text_from_timed_units(tu_file, tu_start, tu_end):
    """Extract text from timed-units XML for range [tu_start, tu_end]."""
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
    """Extract (text, da) for all moves."""
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

def main():
    BASE = "../maptask/maptaskv2-1/Data"  # TODO: download MapTask from https://groups.inf.ed.ac.uk/maptask/ and unzip here
    moves_dir = f"{BASE}/moves"
    tu_dir = f"{BASE}/timed-units"

    moves_files = sorted([f for f in os.listdir(moves_dir) if f.endswith(".moves.xml")])

    all_utterances = []
    for mf in moves_files:
        parts = mf.replace(".moves.xml", "").split(".")
        if len(parts) < 2:
            continue
        conv_id = parts[0]
        speaker_code = parts[1]

        moves_file = f"{moves_dir}/{mf}"
        tu_file = f"{tu_dir}/{conv_id}.{speaker_code}.timed-units.xml"

        if not os.path.exists(tu_file):
            continue

        try:
            utterances = extract_utterances(moves_file, tu_file)
            for u in utterances:
                u["conv_id"] = conv_id
                u["speaker"] = speaker_code
            all_utterances.extend(utterances)
            if len(utterances) > 0:
                print(f"{mf}: {len(utterances)} utterances")
        except Exception as e:
            print(f"Error on {mf}: {e}")

    print(f"\nTotal: {len(all_utterances)} utterances")

    if all_utterances:
        da_counter = Counter(u["da"] for u in all_utterances)
        print(f"DA distribution (all {len(da_counter)} types):")
        for da, cnt in da_counter.most_common():
            print(f"  {da}: {cnt}")
        print(f"\nSample utterances:")
        for u in all_utterances[:10]:
            print(f"  [{u['da']:12s}] {u['text'][:80]}")

        out_file = "../maptask/maptask_text_da.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_utterances, f, ensure_ascii=False)
        print(f"\nSaved to {out_file}")

if __name__ == "__main__":
    main()
