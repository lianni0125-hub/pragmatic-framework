# Pragmatic Framework

> :brain: Map conversational language into a **geometric space** through dialogue-act supervision

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Framework](https://img.shields.io/badge/Framework-DistilBERT-red.svg)

Turn raw utterances into **768-dimensional pragmatic vectors** where dialogue acts become spatial structure. The framework gives you a trained DistilBERT DA tagger (293-class DAMSL) plus a full analysis toolkit.

### :star: What You Get

| Feature | Description |
|---------|-------------|
| :brain: **DA Plugin** | Pre-trained DistilBERT tagger — predict 293 DAMSL dialogue acts on any English dialogue |
| :chart_with_upwards_trend: **Pragmatic Vectors** | 768-dim vectors capturing communicative function, not just semantics |
| :microscope: **Geometric Analysis** | PCA centroids, cosine similarity, interpolation paths between dialogue acts |
| :busts_in_silhouette: **Speaker Profiling** | Role-based profiling (Guide vs Follower) |
| :arrows_counterclockwise: **Cross-Corpus Generalization** | Trained on SwDA → evaluated on MapTask |

> **"That's interesting." (statement) vs "That's interesting?" (skeptical response)** — same words, **different pragmatic vectors** because they serve different communicative functions.

---

## :wrench: Setup

```bash
# 1. Clone the repo
git clone https://github.com/lianni0125-hub/pragmatic-framework.git
cd pragmatic-framework

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the DA Plugin model from HuggingFace Hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Anni0125/pragmatic-framework', local_dir='plugin')"

# 4. Download MapTask data (required for some scripts)
bash scripts/download_maptask.sh
```

> :bulb: **Chinese user?** If HuggingFace is slow, set a mirror first:
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com    # Linux/macOS
> set HF_ENDPOINT=https://hf-mirror.com       # Windows
> ```

---

## :rocket: Quick Start

```bash
# DA distribution & transition analysis
python scripts/analysis_da_distribution.py

# Plugin accuracy on SwDA
python scripts/analysis_plugin_evaluation.py

# Speaker profiling (requires MapTask data)
python scripts/analysis_speaker_profiles.py

# Cross-corpus evaluation (SwDA → MapTask, requires MapTask data)
python scripts/analysis_cross_corpus.py

# MapTask speaker/topic analysis (requires MapTask data)
python scripts/analysis_maptask_profiles.py

# Geometric space analysis
python scripts/analysis_pragmatic_space.py

# Extract pragmatic vectors from your own data
python scripts/extract_vectors.py --input your_data.jsonl --output your_vectors.pt
```

---

## :package: What's Inside

| Directory | Description |
|-----------|-------------|
| `plugin/` :brain: | Pre-trained DistilBERT DA tagger (293-class DAMSL scheme) |
| `vectors/` :chart_with_upwards_trend: | Pre-computed 768-dim pragmatic vectors |
| `data/` :file_folder: | Pre-processed SwDA train/valid/test splits (JSONL) |
| `scripts/` :scroll: | 14 analysis & utility scripts |
| `figures/` :art: | Output figures (auto-created on first run) |

---

## :bulb: Key Numbers

The framework revolves around three numbers:

| Number | Meaning | Where It's Used |
|--------|---------|----------------|
| **768** | DistilBERT hidden dimension — the vector size for all pragmatic representations | `vectors/prag_vectors.pt`, plugin output |
| **293** | Number of fine-grained SwDA DAMSL tags the plugin can predict | Plugin classification head (768 → 293) |
| **8** | Coarse-grained DA categories for analysis convenience (293 → 8 mapping) | Used in downstream analysis scripts |

> **Why 768?** The plugin is built on DistilBERT, whose encoder outputs 768-dim hidden states. These vectors encode communicative function — two utterances with the same words but different dialogue acts (e.g., "You should go." vs "Should you go?") will have **different** 768-dim pragmatic vectors even though their semantic content is similar.

---

## :gear: Plugin Usage

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
    vector = outputs.logits  # torch.Size([1, 768])
```

---

## :arrow_down: External Data

Some scripts require additional data:

| Data | How to Get | Used By |
|------|-----------|---------|
| **MapTask v2.1** | `bash scripts/download_maptask.sh` (auto-downloads) | `analysis_cross_corpus.py`, `analysis_maptask_profiles.py`, `analysis_speaker_profiles.py` |
| **SwDA transcripts** | Download from [switchboard.co.uk](https://www.switchboard.co.uk/#data), place as `swda.zip` | `analysis_speaker_profiles_from_swda.py` |

---

## :scroll: Scripts Overview

| Script | Description |
|--------|-------------|
| `download_maptask.sh` | Download & process MapTask corpus |
| `extract_vectors.py` | Extract pragmatic vectors from any JSONL data |
| `build_episodes.py` | Build conversation episodes from test data |
| `make_contexts_from_swda.py` | Build context file for vector precomputation |
| `precompute_prag_vectors.py` | Precompute vectors from context file |
| `analysis_da_distribution.py` | DA distribution & transition analysis |
| `analysis_speaker_profiles.py` | Speaker profiles from MapTask episodes |
| `analysis_speaker_profiles_from_swda.py` | Speaker profiles from SwDA zip |
| `analysis_pragmatic_space.py` | Geometric structure of pragmatic space |
| `analysis_plugin_evaluation.py` | Plugin accuracy on SwDA |
| `analysis_cross_corpus.py` | Cross-corpus evaluation (SwDA → MapTask) |
| `analysis_maptask_profiles.py` | MapTask speaker/topic analysis |
| `supplementary_analysis.py` | Confusion matrix, interpolation, multi-label |
| `da_tag_outputs.py` | Tag your own data with DA predictions |
| `discover_new_prag_tags_ABC.py` | Discover new pragmatic tags via clustering |

---

## :book: Citation

```
TODO: Add citation when paper is published
```

---

## :page_facing_up: License

MIT License
