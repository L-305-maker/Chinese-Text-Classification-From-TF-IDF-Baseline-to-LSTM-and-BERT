# Chinese Text Classification: TF-IDF Baseline, LSTM and BERT

本项目基于中文新闻文本分类数据集，完成从传统机器学习 baseline 到深度学习模型的完整文本分类流程。项目将 **TF-IDF + Logistic Regression**、**LSTM** 和 **BERT** 放在同一数据集上进行训练、评估与对比，重点展示中文文本分类任务中的数据处理、模型构建、训练调参、评估分析与工程化组织能力。

> 当前项目更适合作为一个学习型 / 简历展示型 NLP 项目：既包含传统机器学习方法，也包含深度学习与预训练模型方法，能够体现从 baseline 到复杂模型逐步升级的建模思路。

---

## 1. Project Overview

### 1.1 项目目标

本项目主要完成以下工作：

- 将原始中文新闻文本数据处理为模型可读取的结构化 CSV 文件。
- 构建传统机器学习 baseline：`TF-IDF + Logistic Regression`。
- 构建基于词向量和序列建模的 `LSTM` 文本分类模型。
- 构建基于预训练语言模型的 `BERT` 文本分类模型。
- 使用统一的评估指标比较不同模型在同一任务上的表现。
- 通过 `main.py` 提供统一命令行入口，便于运行数据处理、模型训练与结果复现。

### 1.2 任务类型

- **任务名称**：中文新闻文本分类
- **任务类型**：多分类任务
- **类别数量**：10 类
- **核心目标**：输入一段中文新闻文本，预测其所属新闻类别

---

## 2. Dataset

### 2.1 数据集说明

本项目使用中文新闻文本分类数据集，类别包括：

| 类别 | 说明 |
|---|---|
| 体育 | 体育赛事、运动员、比赛相关新闻 |
| 财经 | 金融、股票、经济、商业相关新闻 |
| 游戏 | 游戏产品、电竞、游戏行业相关新闻 |
| 家居 | 家装、居住、生活方式相关新闻 |
| 时政 | 政策、社会、公共事务相关新闻 |
| 房产 | 房地产、楼市、住房相关新闻 |
| 教育 | 学校、考试、教育行业相关新闻 |
| 时尚 | 服饰、美妆、潮流相关新闻 |
| 科技 | 科技公司、互联网、技术相关新闻 |
| 娱乐 | 明星、影视、综艺相关新闻 |

### 2.2 数据来源

数据集来源：

```text
https://hyper.ai/cn/datasets/9277
```

由于完整数据集较大，本项目仅使用其中部分数据进行训练和实验。仓库中默认不上传原始数据文件和大型模型权重文件，请在本地准备数据后运行数据处理脚本。

### 2.3 原始数据格式

原始数据文件位于：

```text
data/raw/
├─ cnews.train.txt
├─ cnews.val.txt
└─ cnews.test.txt
```

每一行数据格式如下：

```text
<label>\t<text>
```

示例：

```text
体育	中国男篮在本场比赛中表现出色……
财经	今日A股市场震荡上行……
```

数据处理脚本会将原始 `.txt` 文件转换为 CSV 格式，并保存到 `data/processed/` 目录下。

---

## 3. Project Structure

```text
Chinese Text Classification From TF-IDF Baseline to LSTM and BERT/
├─ data/
│  ├─ raw/                         # 原始数据文件，因体积较大不上传 GitHub
│  │  ├─ cnews.train.txt
│  │  ├─ cnews.val.txt
│  │  └─ cnews.test.txt
│  └─ processed/                   # 处理后的 CSV 数据，运行 data_process.py 后生成
│     ├─ train_data.csv
│     ├─ val_data.csv
│     └─ test_data.csv
│
├─ models/                         # 模型权重保存目录，较大文件不上传 GitHub
│  ├─ bert/
│  ├─ lstm/
│  └─ lr_tfidf/
│
├─ outputs/                        # 评估结果、预测结果、可视化图片等输出
│
├─ notebooks/                      # 实验分析或可视化 notebook
│
├─ parameters/                     # 模型参数、指标、标签映射等 JSON 文件
│  ├─ bert/
│  │  ├─ config.json
│  │  ├─ history.json
│  │  ├─ label_map.json
│  │  └─ metrics.json
│  ├─ lr_tfidf/
│  │  ├─ best_params.json
│  │  ├─ config.json
│  │  ├─ label_map.json
│  │  └─ metrics.json
│  └─ lstm/
│     ├─ config.json
│     ├─ history.json
│     ├─ label_map.json
│     ├─ metrics.json
│     └─ vocab.json
│
├─ src/
│  ├─ data_process.py              # 原始 txt 数据转 CSV，读取 processed 数据，构建标签 id
│  ├─ datasets_lstm.py             # LSTM 数据集类、分词、padding、collate_fn 等
│  ├─ datasets_bert.py             # BERT tokenizer 编码和 Dataset 封装
│  ├─ evaluate.py                  # 模型评估与指标计算
│  ├─ lr_train.py                  # TF-IDF + Logistic Regression baseline 训练
│  ├─ train_lstm.py                # LSTM 模型训练流程
│  ├─ train_bert.py                # BERT 模型训练流程
│  ├─ predict.py                   # 单条文本或批量文本预测
│  └─ model_utils.py               # 标签映射、通用工具函数等
│
├─ main.py                         # 统一命令行入口
├─ requirements.txt                # 项目依赖
└─ README.md
```

---

## 4. Environment Setup

建议使用 Python 3.9 或以上版本。

### 4.1 创建虚拟环境

```bash
conda create -n text-classification python=3.9
conda activate text-classification
```

或者使用 `venv`：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 4.2 安装依赖

```bash
pip install -r requirements.txt
```

常见依赖包括：

```text
pandas
numpy
scikit-learn
torch
transformers
matplotlib
seaborn
tqdm
joblib
```

> 如果使用 GPU 训练 BERT 或 LSTM，请根据自己的 CUDA 版本安装对应的 PyTorch 版本。

---

## 5. How to Run

### 5.1 数据预处理

将原始数据放入：

```text
data/raw/
```

然后运行：

```bash
python main.py --mode "process data"
```

或直接运行：

```bash
python src/data_process.py
```

处理完成后，将生成：

```text
data/processed/train_data.csv
data/processed/val_data.csv
data/processed/test_data.csv
```

CSV 文件包含两列：

| column | meaning |
|---|---|
| text | 新闻文本内容 |
| label | 新闻类别标签 |

### 5.2 训练 TF-IDF + Logistic Regression baseline

```bash
python main.py --mode "LogisticRegressor + TF-IDF"
```

该模型作为 baseline，用于快速建立可解释、训练成本较低的传统机器学习对照组。

### 5.3 训练 LSTM 模型

```bash
python main.py --mode "LSTM" --epoches 10 --batch_size 64
```

LSTM 模型会将文本转换为 token id 序列，并通过 embedding 层、LSTM 层和分类层完成文本分类。

### 5.4 训练 BERT 模型

```bash
python main.py --mode "BERT" --epoches 3 --batch_size 16
```

BERT 模型基于 `bert-base-chinese`，使用 Hugging Face Transformers 提供的 tokenizer 和预训练模型完成中文文本编码与分类。

> BERT 对显存和训练时间要求较高。如果本地设备性能有限，可以减少 batch size、减少训练轮数，或优先在较小数据子集上验证代码流程。

---

## 6. Methodology

### 6.1 Data Processing

数据处理流程包括：

1. 读取 `data/raw/` 下的 `cnews.train.txt`、`cnews.val.txt` 和 `cnews.test.txt`。
2. 按行读取原始文本。
3. 去除空行。
4. 使用 `\t` 将每一行切分为 `label` 和 `text`。
5. 将文本和标签保存为 DataFrame。
6. 输出为 UTF-8-SIG 编码的 CSV 文件。
7. 在训练阶段读取 `data/processed/` 下的 CSV 文件。
8. 使用统一的 `LABEL2ID` 将中文标签转换为数字类别 id。

这样做的好处是：

- 原始数据处理逻辑清晰，便于检查。
- 所有模型都可以复用同一份 processed 数据。
- 标签映射统一，避免不同模型之间类别 id 不一致。
- 数据处理、模型训练、结果评估之间职责分离，项目结构更清晰。

### 6.2 TF-IDF + Logistic Regression

该模型是本项目的 baseline。

核心流程：

1. 使用 TF-IDF 将文本转换为稀疏向量。
2. 使用 Logistic Regression 完成多分类预测。
3. 使用 Pipeline 将特征提取和分类模型串联。
4. 使用 GridSearchCV 对关键参数进行搜索。
5. 保存最佳参数、模型结果和评估指标。

优点：

- 训练速度快。
- 结果稳定。
- 可解释性较强。
- 适合作为后续深度学习模型的对照组。

局限：

- 主要依赖词频统计，难以建模复杂上下文语义。
- 对语序、长距离依赖和一词多义的处理能力有限。

### 6.3 LSTM Classifier

LSTM 模型用于捕捉文本序列中的上下文信息。

核心流程：

1. 对文本进行分词或字符级切分。
2. 构建词表 `vocab`。
3. 将 token 转换为 token id。
4. 使用 `collate_fn` 对 batch 内不同长度文本进行 padding。
5. 使用真实长度 `lengths` 配合 `pack_padded_sequence` 处理变长序列。
6. 将序列输入 embedding 层和 LSTM 层。
7. 使用最后的隐状态或池化后的序列表示进行分类。
8. 使用交叉熵损失函数训练模型。

优点：

- 能够建模文本的顺序信息。
- 相比传统机器学习模型，具备更强的上下文建模能力。
- 适合学习深度学习文本分类中的 embedding、序列建模和 batch padding 机制。

局限：

- 训练速度慢于 TF-IDF baseline。
- 对长文本的建模能力仍然有限。
- 效果受分词方式、词表大小、序列长度、hidden size 等参数影响较大。

### 6.4 BERT Classifier

BERT 模型使用预训练语言模型进行中文文本分类。

核心流程：

1. 使用 `bert-base-chinese` tokenizer 对文本进行编码。
2. 得到 `input_ids`、`attention_mask` 等输入。
3. 将编码后的 batch 输入 BERT。
4. 使用 `[CLS]` token 对应的表示或模型输出进行分类。
5. 使用交叉熵损失函数进行微调。
6. 在验证集和测试集上评估模型效果。

优点：

- 利用大规模语料预训练得到的语言知识。
- 对上下文语义、词义变化和复杂文本关系建模能力更强。
- 通常在中文文本分类任务中表现较好。

局限：

- 训练成本较高。
- 对显存要求较高。
- 推理速度慢于传统机器学习模型。

---

## 7. Evaluation

本项目使用多个指标评估模型表现：

| 指标 | 含义 | 适用原因 |
|---|---|---|
| Accuracy | 预测正确样本数占总样本数比例 | 直观衡量整体分类正确率 |
| Precision | 被预测为某类的样本中有多少是真的 | 适合分析误报情况 |
| Recall | 某一真实类别中有多少被正确找回 | 适合分析漏报情况 |
| F1-score | Precision 和 Recall 的综合指标 | 适合类别分布不完全均衡的多分类任务 |
| Confusion Matrix | 各类别之间的预测混淆情况 | 适合观察哪些类别容易被模型混淆 |

建议在 `outputs/` 中保存：

```text
outputs/
├─ model_comparison.csv
├─ model_comparison.png
├─ confusion_matrix_lr.png
├─ confusion_matrix_lstm.png
└─ confusion_matrix_bert.png
```

模型对比表可以采用如下形式：

| Model | Accuracy | Macro F1 | Weighted F1 | Training Cost | Notes |
|---|---:|---:|---:|---|---|
| TF-IDF + Logistic Regression | 待填写 | 待填写 | 待填写 | Low | baseline，速度快，可解释性强 |
| LSTM | 待填写 | 待填写 | 待填写 | Medium | 能建模序列信息，但对参数较敏感 |
| BERT | 待填写 | 待填写 | 待填写 | High | 语义建模能力强，训练成本较高 |

> 建议在完成训练后，将 `parameters/*/metrics.json` 中的最终结果填入该表格，增强 README 的可信度。

---

## 8. Project Highlights

- **完整 NLP 建模流程**：包含数据清洗、文本向量化、模型训练、模型评估和结果保存。
- **baseline 思维明确**：先使用 `TF-IDF + Logistic Regression` 建立传统机器学习对照组，再与深度学习模型比较。
- **模型层次递进**：从词频统计方法，到 LSTM 序列模型，再到 BERT 预训练模型，体现模型复杂度逐步提升的实验设计。
- **工程化结构清晰**：使用 `src/`、`parameters/`、`models/`、`outputs/` 等目录分离代码、参数、模型和结果。
- **统一入口运行**：通过 `main.py` 和 `argparse` 管理不同运行模式，便于命令行复现。
- **可复现实验记录**：使用 JSON 文件保存 config、metrics、history、label map 等信息。
- **多指标评估**：不仅关注 accuracy，也关注 F1-score 和混淆矩阵，避免只用单一指标判断模型效果。

---

## 9. Possible Improvements

后续可以继续优化：

- 增加模型结果可视化，例如训练 loss 曲线、验证集 F1 曲线、模型对比柱状图。
- 增加混淆矩阵，分析不同类别之间的误判关系。
- 对 BERT 增加学习率、max length、batch size 等参数实验。
- 对 LSTM 增加双向 LSTM、dropout、attention pooling 等改进。
- 增加 `predict.py` 的命令行预测功能，支持输入一条中文文本并输出预测类别。
- 增加错误样本分析，展示模型在哪些文本上容易预测错误。
- 增加 `README` 中的最终实验结果表和关键可视化图片。

---

## 10. Notes for GitHub

由于数据集和模型权重文件较大，建议不要直接上传以下内容：

```text
data/raw/
data/processed/
models/
*.pth
*.pt
*.bin
```

可以在 `.gitignore` 中加入：

```gitignore
data/raw/
data/processed/
models/
*.pth
*.pt
*.bin
__pycache__/
.ipynb_checkpoints/
```

但建议保留以下文件：

```text
parameters/*/config.json
parameters/*/metrics.json
parameters/*/label_map.json
outputs/model_comparison.png
outputs/confusion_matrix_*.png
```

这些文件可以帮助别人快速了解你的实验设置和最终结果。

---

## 11. Summary

本项目完成了一个较完整的中文文本分类实验：从原始文本数据处理开始，依次构建传统机器学习 baseline、LSTM 深度学习模型和 BERT 预训练模型，并使用统一指标对不同模型进行评估与比较。

