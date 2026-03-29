#!/usr/bin/env python3
"""Extract pragmatic vectors from text using the DA Plugin."""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def extract_vectors(input_file, output_file, plugin_dir="../plugin"):
    tokenizer = AutoTokenizer.from_pretrained(plugin_dir)
    model = AutoModelForSequenceClassification.from_pretrained(plugin_dir)
    model.eval()

    vectors = []
    labels = []
    texts = []

    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            label = item.get("label", None)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                outputs = model(**inputs)
                vec = outputs.logits.squeeze().cpu().numpy()
            vectors.append(vec)
            texts.append(text)
            if label is not None:
                labels.append(label)

    result = {"vectors": vectors, "texts": texts, "labels": labels}
    torch.save(result, output_file)
    print(f"Saved {len(vectors)} vectors to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--plugin", default="../plugin", help="Plugin directory")
    args = parser.parse_args()
    extract_vectors(args.input, args.output, args.plugin)
