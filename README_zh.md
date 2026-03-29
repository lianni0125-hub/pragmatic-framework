# 🎯 Pragmatic Framework

> 一个基于对话行为监督的会话语言几何分析框架 ✨

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📖 概述

本仓库提供了一个基于 **DistilBERT 的对话行为（DA）标注器（Plugin）**，该模型在 **Switchboard Dialog Act Corpus (SwDA)** 上训练。此外还提供了由 Plugin 输出的 **768 维语用向量**集合，可用于会话语言语用功能的几何分析。

## 🎯 资源一览

| 资源 | 描述 | 大小 |
|------|------|------|
| 📦 `plugin/` | 预训练 DA Plugin 模型 | ~255 MB |
| 🧮 `vectors/` | 预计算的 768 维语用向量 | ~300 KB |
| 📊 `data/` | 标注数据集划分（train/valid/test） | ~5.9 MB |
| 🐍 `scripts/` | 可复现的分析脚本 | — |

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/lianni0125-hub/pragmatic-framework.git
cd pragmatic-framework
pip install -r requirements.txt
```

### 提取语用向量

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
    vector = outputs.logits  # 🎯 768 维语用向量
```

### 复现论文分析

```bash
# PCA 可视化与插值分析
python scripts/pca_interpolation.py --vectors vectors/prag_vectors.pt

# 语料库级分布分析
python scripts/da_distribution.py

# 跨语料库评估（SwDA → MapTask）
python scripts/cross_corpus_maptask.py
```

## 📁 项目结构

```
pragmatic-framework/
├── plugin/                          # 🎯 预训练 DA Plugin
│   ├── model.safetensors            # 模型权重
│   ├── tokenizer.json               # 分词器
│   ├── vocab.txt                    # 词表
│   ├── label_map.json               # SwDA 标签映射
│   ├── special_tokens_map.json      # 特殊 token 映射
│   ├── tokenizer_config.json        # 分词器配置
│   └── training_args.bin            # 训练参数
├── vectors/                         # 🧮 语用向量
│   └── prag_vectors.pt              # 所有话语 → 768 维向量
├── data/                            # 📊 数据集划分（SwDA）
│   ├── train.jsonl                  # 训练集
│   ├── valid.jsonl                  # 验证集
│   └── test.jsonl                   # 测试集
├── maptask/                          # 🗂️ HCRC MapTask 语料（用于跨语料库评估）
│   ├── maptaskv2-1/                 # ⚠️ 从 https://groups.inf.ed.ac.uk/maptask/ 下载并解压到此处
│   ├── maptask_text_da.json         # 提取的 MapTask 话语及 DA 标签
│   └── episodes_T6.jsonl             # 对话片段（每段6轮）
├── swda.zip                          # 完整 SwDA 转录 zip（用于 speaker 画像）
├── scripts/                         # 🐍 分析脚本
│   ├── coarse_label_mapping.py     # SwDA 43类到7粗粒度类别映射
│   ├── cross_corpus_maptask.py     # Plugin 在 MapTask 语料上的评估
│   ├── da_distribution.py           # DA 分布 + 转移矩阵
│   ├── extract_maptask_text.py     # 从 MapTask XML 提取文本和 DA
│   ├── extract_vectors.py           # 从文本提取语用向量
│   ├── pca_interpolation.py         # 语用空间可视化（PCA）+ 插值
│   ├── plugin_prediction_distribution.py  # 混淆矩阵 + 插值分析
│   ├── precompute_prag_vectors.py   # 预计算所有话语的语用向量
│   ├── speaker_distribution.py     # 按 speaker 统计 DA 分布
│   ├── speaker_profiling.py         # 从 SwDA zip 提取 speaker 画像
│   └── topic_da_analysis.py         # MapTask Topic-DA 分布
├── requirements.txt
└── README.md
```

## 📋 依赖

- Python 3.8+
- PyTorch 1.10+
- transformers
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- umap-learn（可选，用于 UMAP 可视化）

## 🔬 这个框架能做什么？

- 🔍 **语料库级分布分析** — 可视化对话行为在几何空间中的聚类情况
- 👤 **说话人画像** — 根据说话人在语用空间中的位置对个体进行画像
- 📈 **序列动态分析** — 分析对话流程中对话行为之间的转移模式
- 🌐 **跨语料库泛化** — Plugin 在 SwDA 上训练，MapTask 上合作类别准确率 70.7%

## 📄 许可证

本项目采用 **MIT 许可证**。

## 📢 引用

```
TODO: 论文发表后添加引用信息
```
