# 🎯 Pragmatic Framework

> A geometric framework for analyzing conversational language through dialogue-act supervision ✨

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📖 Overview

This repository provides a **DistilBERT-based Dialogue Act (DA) tagger (Plugin)** trained on the **Switchboard Dialog Act Corpus (SwDA)**, along with a set of **768-dimensional pragmatic vectors** derived from the Plugin's output. These resources enable geometric analysis of pragmatic functions in conversational language.

## 🎯 Resources at a Glance

| Resource | Description | Size |
|----------|-------------|------|
| 📦 `plugin/` | Pre-trained DA Plugin model | ~255 MB |
| 🧮 `vectors/` | Pre-computed 768-dim pragmatic vectors | ~300 KB |
| 📊 `data/` | Annotated dataset splits (train/valid/test) | ~5.9 MB |
| 🐍 `scripts/` | Reproducible analysis scripts | — |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/lianni0125-hub/pragmatic-framework.git
cd pragmatic-framework
pip install -r requirements.txt
```

### Download Model (required for running the Plugin)

The model file (`model.safetensors`, ~255 MB) is too large for GitHub and must be downloaded separately:

```bash
# Option 1: Download from HuggingFace Hub (recommended)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='lianni0125/pragmatic-plugin', filename='model.safetensors', local_dir='plugin')"

# Option 2: Download manually
# Visit https://huggingface.co/lianni0125/pragmatic-plugin, download model.safetensors, and place it in the plugin/ directory
```

### Download MapTask Corpus (optional, for cross-corpus evaluation)

```bash
# Visit https://groups.inf.ed.ac.uk/maptask/
# Download maptaskv2-1.tar.gz and unzip to maptask/maptaskv2-1/
```

### Extract Pragmatic Vectors

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

plugin_dir = "plugin"
tokenizer = AutoTokenizer.from_pretrained(plugin_dir)
model = AutoModelForSequenceClassification.from_pretrained(plugin_dir)

text = "Yeah I think that would be great"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
with torch.no_grad():
    outputs = model(**inputs)
    vector = outputs.logits  # 🎯 768-dim pragmatic vector
```

### Reproduce Paper Analyses

```bash
# PCA visualization & interpolation analysis
python scripts/pca_interpolation.py --vectors vectors/prag_vectors.pt

# Corpus-level distribution analysis
python scripts/da_distribution.py

# Cross-corpus evaluation (SwDA → MapTask)
python scripts/cross_corpus_maptask.py
```

## 📁 Project Structure

```
pragmatic-framework/
├── plugin/                          # 🎯 Pre-trained DA Plugin
│   ├── model.safetensors            # ⚠️ Download from HuggingFace: https://huggingface.co/lianni0125/pragmatic-plugin
│   ├── tokenizer.json               # Tokenizer
│   ├── vocab.txt                    # Vocabulary
│   ├── label_map.json               # SwDA label mapping
│   ├── special_tokens_map.json      # Special tokens mapping
│   ├── tokenizer_config.json        # Tokenizer configuration
│   └── training_args.bin            # Training arguments
├── vectors/                         # 🧮 Pragmatic vectors
│   └── prag_vectors.pt              # All utterances → 768-d vectors
├── data/                            # 📊 Dataset splits (SwDA)
│   ├── train.jsonl                  # Training set
│   ├── valid.jsonl                  # Validation set
│   └── test.jsonl                   # Test set
├── maptask/                          # 🗂️ HCRC MapTask corpus (for cross-corpus eval)
│   ├── maptaskv2-1/                 # ⚠️ Download from: https://groups.inf.ed.ac.uk/maptask/ and place here
│   ├── maptask_text_da.json         # Extracted MapTask utterances with DA labels
│   └── episodes_T6.jsonl             # Dialog episodes (6 turns each)
├── swda.zip                          # Full SwDA transcript zip (for speaker profiling)
├── scripts/                         # 🐍 Analysis scripts
│   ├── coarse_label_mapping.py     # Map SwDA 43-class to 7 coarse categories
│   ├── cross_corpus_maptask.py     # Evaluate Plugin on MapTask corpus
│   ├── da_distribution.py           # DA distribution + transition matrix
│   ├── extract_maptask_text.py     # Extract text + DA from MapTask XML
│   ├── extract_vectors.py          # Extract pragmatic vectors from text
│   ├── pca_interpolation.py         # Visualize pragmatic space (PCA) + interpolation
│   ├── plugin_prediction_distribution.py  # Confusion matrix + interpolation
│   ├── precompute_prag_vectors.py   # Precompute vectors for all utterances
│   ├── speaker_distribution.py     # DA distribution by speaker
│   ├── speaker_profiling.py         # Speaker profiling from SwDA zip
│   └── topic_da_analysis.py         # MapTask topic-DA distribution
├── requirements.txt
└── README.md
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.10+
- transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- umap-learn (optional, for UMAP visualization)

## 🔬 What Can You Do With It?

- 🔍 **Corpus-level distribution analysis** — Visualize how dialogue acts cluster in geometric space
- 👤 **Speaker-level profiling** — Profile individual speakers by their position in the pragmatic space
- 📈 **Sequential dynamics** — Analyze transitions between dialogue acts in conversation flow
- 🌐 **Cross-corpus generalization** — Plugin trained on SwDA, evaluated on MapTask (cooperative categories: 70.7% accuracy)

## 📄 License

This project is licensed under the **MIT License**.

## 📢 Citation

```
TODO: Add citation when paper is published
```
