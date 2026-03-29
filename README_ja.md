# 🎯 Pragmatic Framework

> 対話行為の監視に基づく会話言語の幾何学的分析フレームワーク ✨

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📖 概要

本リポジトリは、**Switchboard Dialog Act Corpus (SwDA)** で学習した **DistilBERT ベースの対話行為（DA）タガー（Plugin）** を提供します。また、Plugin の出力から得られる **768 次元の語用ベクトル** セットを提供し、会話言語の語用機能の幾何学的分析を可能にします。

## 🎯 リソース一覧

| リソース | 説明 | サイズ |
|----------|------|--------|
| 📦 `plugin/` | 事前学習済み DA Plugin モデル | ~255 MB |
| 🧮 `vectors/` | 事前計算済みの 768 次元語用ベクトル | ~300 KB |
| 📊 `data/` | 注釈付きデータセット分割（train/valid/test） | ~5.9 MB |
| 🐍 `scripts/` | 再現可能な分析スクリプト | — |

## 🚀 クイックスタート

### インストール

```bash
git clone https://github.com/lianni0125-hub/pragmatic-framework.git
cd pragmatic-framework
pip install -r requirements.txt
```

### モデルのダウンロード（Plugin の実行に必須）

モデルファイル（`model.safetensors`、約 255 MB）は GitHub にアップロードできないため別途ダウンロードが必要です：

```bash
# 方法1：HuggingFace Hub からダウンロード（推奨）
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='lianni0125/pragmatic-plugin', filename='model.safetensors', local_dir='plugin')"

# 方法2：手動ダウンロード
# https://huggingface.co/lianni0125/pragmatic-plugin にアクセスし、model.safetensors をダウンロードして plugin/ ディレクトリに配置
```

### MapTask コーパスのダウンロード（オプション、クロスコーパス評価用）

```bash
# https://groups.inf.ed.ac.uk/maptask/ にアクセス
# maptaskv2-1.tar.gz をダウンロードし、maptask/maptaskv2-1/ に解凍
```

### 語用ベクトルの抽出

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
    vector = outputs.logits  # 🎯 768 次元語用ベクトル
```

### 論文の分析を再現

```bash
# PCA 可視化と補間分析
python scripts/pca_interpolation.py --vectors vectors/prag_vectors.pt

# コーパスレベルの分布分析
python scripts/da_distribution.py

# クロスコーパス評価（SwDA → MapTask）
python scripts/cross_corpus_maptask.py
```

## 📁 プロジェクト構造

```
pragmatic-framework/
├── plugin/                          # 🎯 事前学習済み DA Plugin
│   ├── model.safetensors            # ⚠️ HuggingFace からダウンロード: https://huggingface.co/lianni0125/pragmatic-plugin
│   ├── tokenizer.json               # トークナイザー
│   ├── vocab.txt                    # 語彙
│   ├── label_map.json               # SwDA ラベルマッピング
│   ├── special_tokens_map.json      # 特殊トークンマッピング
│   ├── tokenizer_config.json        # トークナイザー設定
│   └── training_args.bin            # 学習引数
├── vectors/                         # 🧮 語用ベクトル
│   └── prag_vectors.pt              # 全発話 → 768 次元ベクトル
├── data/                            # 📊 データセット分割（SwDA）
│   ├── train.jsonl                  # 訓練セット
│   ├── valid.jsonl                  # 検証セット
│   └── test.jsonl                   # テストセット
├── maptask/                          # 🗂️ HCRC MapTask コーパス（クロスコーパス評価用）
│   ├── maptaskv2-1/                 # ⚠️ https://groups.inf.ed.ac.uk/maptask/ からダウンロードして配置
│   ├── maptask_text_da.json         # 抽出した MapTask 発話と DA ラベル
│   └── episodes_T6.jsonl             # 対話セグメント（各6ターン）
├── swda.zip                          # 完全 SwDA 書き起こし zip（話者プロファイリング用）
├── scripts/                         # 🐍 分析スクリプト
│   ├── coarse_label_mapping.py     # SwDA 43クラスから7粗粒度カテゴリへのマッピング
│   ├── cross_corpus_maptask.py     # Plugin の MapTask コーパスでの評価
│   ├── da_distribution.py           # DA 分布 + 遷移行列
│   ├── extract_maptask_text.py     # MapTask XML からテキストと DA を抽出
│   ├── extract_vectors.py           # テキストから語用ベクトルを抽出
│   ├── pca_interpolation.py         # 語用空間の可視化（PCA）+ 補間
│   ├── plugin_prediction_distribution.py  # 混同行列 + 補間分析
│   ├── precompute_prag_vectors.py   # 全発話の語用ベクトルを事前計算
│   ├── speaker_distribution.py     # 話者ごとの DA 分布統計
│   ├── speaker_profiling.py         # SwDA zip から話者プロファイルを抽出
│   └── topic_da_analysis.py         # MapTask Topic-DA 分布
├── requirements.txt
└── README.md
```

## 📋 依存関係

- Python 3.8+
- PyTorch 1.10+
- transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- umap-learn（オプション、UMAP 可視化用）

## 🔬 このフレームワークでできること

- 🔍 **コーパスレベルの分布分析** — 対話行為が幾何学的空間にクラスタリングされる様子を可視化
- 👤 **話者レベルのプロファイリング** — 話者の語用空間における位置に基づいて個人をプロファイリング
- 📈 **逐次動態分析** — 会話の流れにおける対話行為間の遷移パターンを分析
- 🌐 **クロスコーパス汎化** — SwDA で学習した Plugin を MapTask で評価（協調カテゴリ精度 70.7%）

## 📄 ライセンス

本プロジェクトは **MIT ライセンス** を採用しています。

## 📢 引用

```
TODO: 論文発表後に引用情報を追加
```
