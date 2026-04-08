# Pragmatic Framework

> :brain: 对话行为监督让语言在**几何空间**中呈现清晰结构

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Framework](https://img.shields.io/badge/Framework-DistilBERT-red.svg)

将原始对话文本转化为 **768 维语用向量**，对话行为在空间中呈现出清晰的结构。框架提供预训练的 DistilBERT DA 标注器（293 类 DAMSL）以及完整的分析工具链。

### :star: 核心亮点

| 功能 | 说明 |
|------|------|
| :brain: **DA Plugin** | 预训练 DistilBERT 标注器 — 可对任意英语对话预测 293 类 DAMSL 对话行为 |
| :chart_with_upwards_trend: **语用向量** | 768 维向量，捕捉交流功能而非仅语义内容 |
| :microscope: **几何分析** | PCA 中心点、余弦相似度、对话行为间的插值路径 |
| :busts_in_silhouette: **说话人画像** | 基于角色的画像（Guide vs Follower） |
| :arrows_counterclockwise: **跨语料泛化** | 在 SwDA 上训练 → 在 MapTask 上评测 |

> **"That's interesting."（陈述）vs "That's interesting?"（质疑回应）** — 词相同，但**交流功能不同** → pragmatic 向量不同。

---

## :wrench: 安装

```bash
# 1. 克隆仓库
git clone https://github.com/lianni0125-hub/pragmatic-framework.git
cd pragmatic-framework

# 2. 安装依赖
pip install -r requirements.txt

# 3. 从 HuggingFace Hub 下载 DA Plugin 模型
python -c "from huggingface_hub import snapshot_download; snapshot_download('Anni0125/pragmatic-framework', local_dir='plugin')"

# 4. 下载 MapTask 数据（部分脚本需要）
bash scripts/download_maptask.sh
```

> :bulb: **国内用户？** 如果访问 HuggingFace 较慢，先设置镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com    # Linux/macOS
> set HF_ENDPOINT=https://hf-mirror.com       # Windows
> ```

---

## :rocket: 快速开始

```bash
# DA 分布 & 转移分析
python scripts/analysis_da_distribution.py

# Plugin 在 SwDA 上的准确率评估
python scripts/analysis_plugin_evaluation.py

# 说话人画像（需要 MapTask 数据）
python scripts/analysis_speaker_profiles.py

# 跨语料评估（SwDA → MapTask，需要 MapTask 数据）
python scripts/analysis_cross_corpus.py

# MapTask 说话人/话题分析（需要 MapTask 数据）
python scripts/analysis_maptask_profiles.py

# 语用空间几何结构分析
python scripts/analysis_pragmatic_space.py

# 从你自己的数据提取语用向量
python scripts/extract_vectors.py --input your_data.jsonl --output your_vectors.pt
```

---

## :package: 目录结构

| 目录 | 说明 |
|------|------|
| `plugin/` :brain: | 预训练 DistilBERT DA 标注器（293 类 DAMSL 方案） |
| `vectors/` :chart_with_upwards_trend: | 预计算的 768 维语用向量 |
| `data/` :file_folder: | 预处理后的 SwDA train/valid/test 数据（JSONL 格式） |
| `scripts/` :scroll: | 14 个分析 & 工具脚本 |
| `figures/` :art: | 输出图片目录（首次运行自动创建） |

---

## :bulb: 三个关键数字

整个框架围绕三个数字展开：

| 数字 | 含义 | 用途 |
|------|------|------|
| **768** | DistilBERT 隐藏层维度，即所有语用表征的向量大小 | `vectors/prag_vectors.pt`、plugin 输出 |
| **293** | Plugin 可以预测的 SwDA DAMSL 细粒度标签数量 | Plugin 分类头（768 → 293） |
| **8** | 为方便分析将 293 类映射为 8 个粗粒度类别 | 下游分析脚本使用 |

> **为什么是 768？** Plugin 基于 DistilBERT 构建，其编码器输出 768 维隐藏状态。这些向量编码的是交流功能——两个词相同但对话行为不同的句子（例如 "You should go." vs "Should you go?"）会有**不同**的 768 维语用向量，尽管它们的语义内容相似。

---

## :gear: Plugin 使用方法

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

## :arrow_down: 外部数据

部分脚本需要额外数据：

| 数据 | 获取方式 | 使用脚本 |
|------|---------|---------|
| **MapTask v2.1** | `bash scripts/download_maptask.sh`（自动下载） | `analysis_cross_corpus.py`、`analysis_maptask_profiles.py`、`analysis_speaker_profiles.py` |
| **SwDA 转写文本** | 从 [switchboard.co.uk](https://www.switchboard.co.uk/#data) 下载，放入 `swda.zip` | `analysis_speaker_profiles_from_swda.py` |

---

## :scroll: 脚本一览

| 脚本 | 说明 |
|------|------|
| `download_maptask.sh` | 下载并处理 MapTask 语料 |
| `extract_vectors.py` | 从任意 JSONL 数据提取语用向量 |
| `build_episodes.py` | 从测试数据构建对话片段 |
| `make_contexts_from_swda.py` | 为向量预计算构建上下文文件 |
| `precompute_prag_vectors.py` | 从上下文文件预计算语用向量 |
| `analysis_da_distribution.py` | DA 分布 & 转移分析 |
| `analysis_speaker_profiles.py` | 从 MapTask episodes 分析说话人画像 |
| `analysis_speaker_profiles_from_swda.py` | 从 SwDA zip 分析说话人画像 |
| `analysis_pragmatic_space.py` | 语用空间几何结构分析 |
| `analysis_plugin_evaluation.py` | Plugin 在 SwDA 上的准确率评估 |
| `analysis_cross_corpus.py` | 跨语料评估（SwDA → MapTask） |
| `analysis_maptask_profiles.py` | MapTask 说话人/话题分析 |
| `supplementary_analysis.py` | 混淆矩阵、插值、多标签分析 |
| `da_tag_outputs.py` | 用 DA 预测标注你自己的数据 |
| `discover_new_prag_tags_ABC.py` | 通过聚类发现新语用标签 |

---

## :book: 引用

```
TODO: 论文发表后添加引用信息
```

---

## :page_facing_up: 许可证

MIT License
