# Chinese Text Classification: TF-IDF, LSTM, BERT, Freeze Sweep and FGM

> 项目结构已整理：源码放在 `src/`，模型定义放在 `src/models/`，训练产物统一放在 `runs/<experiment_name>/`，汇总报告放在 `reports/`，partial 4/8 的 FGM 对比放在 `comparison_fgm/`。新的结构说明见 `docs/PROJECT_STRUCTURE.md`。

本项目基于中文新闻文本分类数据集，完成从传统机器学习 baseline 到深度学习模型、预训练语言模型微调实验的完整流程。当前代码支持：

- 数据预处理：原始 `cnews.*.txt` 转为统一 CSV。
- 传统 baseline：`TF-IDF + Logistic Regression`。
- 深度学习模型：`LSTM` 文本分类器。
- 预训练模型：`bert-base-chinese` 分类器。
- BERT 冻结策略实验：`frozen / partial / full`。
- BERT 可训练参数统计：`trainable ratio`、`trainable parameters`、`total parameters`。
- Partial-4/8 + Embedding Unfrozen + FGM 对抗训练对比。
- 训练曲线、模型对比、混淆矩阵、classification report、预测结果和错例分析输出。

> 说明：README 按当前代码接口重写。命令中的训练轮数参数是 `--epochs`，不是 `--epoches`。

---

## 1. 项目任务

任务类型：中文新闻文本多分类。

输入是一段中文新闻文本，输出是新闻所属类别。当前标签映射定义在 `src/model_utils.py`：

| id | label |
|---:|---|
| 0 | 体育 |
| 1 | 财经 |
| 2 | 娱乐 |
| 3 | 家居 |
| 4 | 房产 |
| 5 | 教育 |
| 6 | 时尚 |
| 7 | 时政 |
| 8 | 游戏 |
| 9 | 科技 |

---

## 2. 项目结构

```text
Chinese Text Classification From TF-IDF Baseline to LSTM and BERT/
├─ main.py
├─ requirements.txt
├─ README.md
├─ data/
│  ├─ raw/
│  │  ├─ cnews.train.txt
│  │  ├─ cnews.val.txt
│  │  ├─ cnews.test.txt
│  │  └─ cnews.vocab.txt
│  └─ processed/
│     ├─ train_data.csv
│     ├─ val_data.csv
│     └─ test_data.csv
├─ models/
│  ├─ bert/
│  │  └─ bert_unfreeze_last_n_layers.py
│  └─ lstm/
│     └─ lstm_model.py
├─ parameters/
│  ├─ lr_tfidf/
│  ├─ lstm/
│  ├─ bert_full_no_fgm/
│  ├─ bert_frozen_no_fgm/
│  ├─ bert_partial_last_4_no_fgm/
│  └─ bert_partial_last_4_embedding_fgm/
├─ outputs/
│  ├─ comparison/
│  ├─ bert_freeze_classification_reports/
│  ├─ bert_freeze_confusion_matrices/
│  ├─ bert_freeze_predictions/
│  └─ bert_freeze_wrong_cases/
└─ src/
   ├─ adversarial.py
   ├─ data_process.py
   ├─ datasets_bert.py
   ├─ datasets_lstm.py
   ├─ evaluate.py
   ├─ lr_train.py
   ├─ model_utils.py
   ├─ train_bert.py
   ├─ train_lstm.py
   └─ visualize.py
```

其中：

- `main.py`：统一命令行入口。
- `src/data_process.py`：读取原始 txt，并生成 processed CSV。
- `src/lr_train.py`：训练 TF-IDF + Logistic Regression。
- `src/train_lstm.py`：训练 LSTM。
- `src/train_bert.py`：训练 BERT，支持冻结策略、参数统计和 FGM。
- `src/adversarial.py`：FGM 对抗训练实现。
- `src/evaluate.py`：classification report、confusion matrix、预测结果、错例保存。
- `src/visualize.py`：模型指标汇总与可视化。
- `parameters/`：保存 config、metrics、history、label map 等 JSON。
- `models/`：保存模型权重、tokenizer 或 sklearn 模型。
- `outputs/`：保存汇总 CSV、图像、报告、预测和错例分析。

---

## 3. 环境安装

建议使用 Python 3.9+。

```bash
conda create -n textcls python=3.10
conda activate textcls
pip install -r requirements.txt
```

依赖见 `requirements.txt`：

```text
jieba
joblib
matplotlib
numpy
pandas
scikit-learn
torch
transformers
argparse
```

如果使用 GPU 训练，请根据本机 CUDA 版本安装匹配的 PyTorch。首次运行 BERT 时，`transformers` 可能需要下载 `bert-base-chinese`，如果本地没有缓存，需要保证网络可访问 Hugging Face。

---

## 4. 数据格式与预处理

原始数据放在：

```text
data/raw/
├─ cnews.train.txt
├─ cnews.val.txt
└─ cnews.test.txt
```

原始文件每行格式：

```text
label<TAB>text
```

运行预处理：

```bash
python main.py --mode data_processor
```

生成：

```text
data/processed/train_data.csv
data/processed/val_data.csv
data/processed/test_data.csv
```

CSV 包含两列：

| column | meaning |
|---|---|
| text | 新闻正文 |
| label | 中文类别标签 |

---

## 5. 统一入口

项目通过 `main.py` 选择不同模式：

```bash
python main.py --mode data_processor
python main.py --mode Log_TF_IDF
python main.py --mode LSTM
python main.py --mode BERT
python main.py --mode visualize
```

`main.py` 会把 `--mode` 之后的剩余参数转发给对应模块。

---

## 6. 训练 TF-IDF + Logistic Regression

快速检查数据：

```bash
python main.py --mode Log_TF_IDF --check-data
```

使用较小参数网格快速训练：

```bash
python main.py --mode Log_TF_IDF --small-grid
```

使用部分训练样本调试：

```bash
python main.py --mode Log_TF_IDF --sample-size 5000 --small-grid
```

完整训练：

```bash
python main.py --mode Log_TF_IDF
```

输出位置：

```text
models/lr_tfidf/model.pkl
parameters/lr_tfidf/config.json
parameters/lr_tfidf/best_params.json
parameters/lr_tfidf/metrics.json
parameters/lr_tfidf/label_map.json
```

实现细节：

- 使用 `jieba` 分词。
- 使用 `TfidfVectorizer` 提取文本特征。
- 使用 `LogisticRegression(solver="saga")` 训练分类器。
- 使用 `GridSearchCV` 搜索 TF-IDF 和 LR 参数。

---

## 7. 训练 LSTM

默认训练：

```bash
python main.py --mode LSTM
```

常用参数：

```bash
python main.py --mode LSTM --epochs 5 --batch_size 16 --embed_dim 64 --hidden_size 512
```

可配置参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--embed_dim` | 64 | embedding 维度 |
| `--hidden_size` | 512 | LSTM hidden size |
| `--num_layers` | 1 | LSTM 层数 |
| `--dropout` | 0.3 | dropout |
| `--batch_size` | 16 | batch size |
| `--learning_rate` | 0.001 | Adam 学习率 |
| `--epochs` | 5 | 训练轮数 |

输出位置：

```text
models/lstm/best_model.pth
parameters/lstm/config.json
parameters/lstm/history.json
parameters/lstm/metrics.json
parameters/lstm/label_map.json
parameters/lstm/vocab.json
```

---

## 8. 训练 BERT

BERT 分类器定义在 `models/bert/bert_unfreeze_last_n_layers.py`，训练逻辑在 `src/train_bert.py`。

### 8.1 全参数微调

```bash
python main.py --mode BERT --finetune_strategy full --epochs 3 --batch_size 8 --max_len 128
```

当前代码会使用实验名保存，full 参数微调默认保存为：

```text
models/bert_full_no_fgm/
parameters/bert_full_no_fgm/
```

### 8.2 冻结 BERT，仅训练分类头

```bash
python main.py --mode BERT --finetune_strategy frozen --epochs 3 --batch_size 8 --max_len 128
```

保存为：

```text
models/bert_frozen_no_fgm/
parameters/bert_frozen_no_fgm/
```

### 8.3 部分解冻最后 N 层

解冻最后 4 层：

```bash
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4 --epochs 3 --batch_size 8 --max_len 128
```

解冻最后 8 层：

```bash
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8 --epochs 3 --batch_size 8 --max_len 128
```

部分解冻实验会保存为：

```text
models/bert_partial_last_4_no_fgm/
parameters/bert_partial_last_4_no_fgm/
```

### 8.4 BERT 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--dropout` | 0.3 | 分类头 dropout |
| `--max_len` | 256 | BERT tokenizer 最大长度 |
| `--batch_size` | 16 | batch size |
| `--epochs` | 5 | 训练轮数 |
| `--finetune_strategy` | `full` | `full` / `frozen` / `partial` |
| `--unfreeze_last_n_layers` | `None` | partial 策略下解冻最后 N 层；未传时默认为 2 |
| `--bert_lr` | `2e-5` | BERT 参数学习率 |
| `--classifier_lr` | `1e-4` | 分类头学习率 |
| `--weight_decay` | `0.01` | AdamW weight decay |

---

## 9. BERT 冻结策略 Sweep

一次性运行多种冻结策略：

```bash
python main.py --mode BERT --freeze-sweep --epochs 3 --batch_size 8 --max_len 128
```

默认会运行：

```text
bert_frozen_no_fgm
bert_partial_last_1_no_fgm
bert_partial_last_2_no_fgm
bert_partial_last_4_no_fgm
bert_partial_last_8_no_fgm
bert_partial_last_12_no_fgm
bert_full_no_fgm
```

指定 partial 层数：

```bash
python main.py --mode BERT --freeze-sweep --partial-unfreeze-layers 1 4 8 12 --epochs 3 --batch_size 8 --max_len 128
```

Sweep 汇总会保存：

```text
outputs/bert_freeze_summary.csv
```

CSV 中包含：

- `experiment_name`
- `run_name`
- `finetune_strategy`
- `unfreeze_last_n_layers`
- `use_fgm`
- `fgm_epsilon`
- `embedding_unfrozen`
- `trainable_ratio`
- `trainable_ratio_percent`
- `trainable_params`
- `total_params`
- `train_loss`
- `train_acc`
- `train_f1`
- `test_loss`
- `test_acc`
- `test_f1`
- `best_val_loss`
- `best_val_acc`
- `best_val_f1`

同时会保存每个实验的：

```text
outputs/bert_freeze_classification_reports/
outputs/bert_freeze_confusion_matrices/
outputs/bert_freeze_predictions/
outputs/bert_freeze_wrong_cases/
```

---

## 10. Partial-4/8 + Embedding Unfrozen + FGM

FGM 实现在 `src/adversarial.py`。当前设计不是简单地对 frozen embedding 临时扰动，而是单独定义了两类实验：

```text
Partial-4 + Embedding Unfrozen + FGM
Partial-8 + Embedding Unfrozen + FGM
```

开启 FGM sweep：

```bash
python main.py --mode BERT --freeze-sweep --use-fgm --fgm-epsilon 1.0 --epochs 3 --batch_size 8 --max_len 128
```

只要传入 `--use-fgm`，即使 `--partial-unfreeze-layers` 没有写 4 或 8，代码也会自动补齐 partial 4/8 的成对对比：

```text
bert_partial_last_4_no_fgm
bert_partial_last_4_embedding_fgm
bert_partial_last_8_no_fgm
bert_partial_last_8_embedding_fgm
```

FGM 只允许在以下策略中使用：

```text
finetune_strategy = partial
unfreeze_last_n_layers in {4, 8}
```

FGM 版本会额外解冻：

```text
bert.embeddings.*
```

因此 `embedding_fgm` 实验中的 `trainable parameters` 和 `trainable ratio` 会真实反映 embedding 参与训练后的参数量。

FGM 对比输出：

```text
comparison_fgm/bert_fgm_comparison.csv
comparison_fgm/bert_fgm_comparison.png
comparison_fgm/bert_fgm_gain.png
```

---

## 11. 可视化

### 11.1 汇总 LR / LSTM / BERT 指标

默认读取 `parameters/` 下各模型的 `metrics.json` 和 `history.json`：

```bash
python main.py --mode visualize
```

也可以指定模型：

```bash
python main.py --mode visualize --models lr_tfidf lstm bert_full_no_fgm
```

输出：

```text
outputs/model_metrics_summary.csv
outputs/model_metrics_summary.json
outputs/model_test_scores.png
outputs/model_metric_heatmap.png
outputs/model_test_loss.png
outputs/model_generalization_gap.png
outputs/training_loss_curves.png
outputs/training_acc_curves.png
outputs/training_f1_curves.png
```

### 11.2 只可视化 BERT 冻结实验

```bash
python main.py --mode visualize --task bert_freeze --bert-freeze-summary outputs/bert_freeze_summary.csv
```

输出：

```text
outputs/bert_freeze_scores.png
outputs/bert_freeze_trainable_params.png
outputs/bert_freeze_generalization_gap.png
outputs/bert_freeze_efficiency.png
comparison_fgm/bert_fgm_comparison.csv
comparison_fgm/bert_fgm_comparison.png
comparison_fgm/bert_fgm_gain.png
```

### 11.3 同时可视化模型对比和 BERT freeze sweep

```bash
python main.py --mode visualize --task both --bert-freeze-summary outputs/bert_freeze_summary.csv
```

---

## 12. 评估与错误分析

`src/evaluate.py` 提供以下能力：

- accuracy / precision / recall / F1 计算。
- `classification_report` 保存为 txt。
- `confusion_matrix` 保存为 png。
- 预测结果保存为 CSV。
- 错误样本保存为 CSV。

BERT freeze sweep 每个实验会保存：

```text
outputs/bert_freeze_classification_reports/{run_name}_classification_report.txt
outputs/bert_freeze_confusion_matrices/{run_name}_confusion_matrix.png
outputs/bert_freeze_predictions/{run_name}_predictions.csv
outputs/bert_freeze_wrong_cases/{run_name}_wrong_cases.csv
```

其中 wrong cases 文件适合继续做 error analysis，例如查看哪些类别更容易混淆、哪些文本长度或主题表达导致误判。

---

## 13. 推荐实验流程

如果从零开始复现实验，建议按下面顺序运行：

```bash
python main.py --mode data_processor
python main.py --mode Log_TF_IDF --small-grid
python main.py --mode LSTM --epochs 5 --batch_size 16
python main.py --mode BERT --finetune_strategy full --epochs 3 --batch_size 8 --max_len 128
python main.py --mode BERT --freeze-sweep --use-fgm --fgm-epsilon 1.0 --epochs 3 --batch_size 8 --max_len 128
python main.py --mode visualize --task both --bert-freeze-summary outputs/bert_freeze_summary.csv
```

如果显存不足，优先降低：

```bash
--batch_size 4
--max_len 128
--epochs 1
```

---

## 14. 输出文件说明

模型与参数：

```text
models/{experiment_name}/best_model.pth
models/{experiment_name}/tokenizer/
parameters/{experiment_name}/config.json
parameters/{experiment_name}/history.json
parameters/{experiment_name}/metrics.json
parameters/{experiment_name}/label_map.json
```

模型对比：

```text
outputs/model_metrics_summary.csv
outputs/model_metrics_summary.json
outputs/model_test_scores.png
outputs/model_metric_heatmap.png
```

BERT freeze sweep：

```text
outputs/bert_freeze_summary.csv
outputs/bert_freeze_scores.png
outputs/bert_freeze_trainable_params.png
outputs/bert_freeze_generalization_gap.png
outputs/bert_freeze_efficiency.png
```

FGM 对比：

```text
comparison_fgm/bert_fgm_comparison.csv
comparison_fgm/bert_fgm_comparison.png
comparison_fgm/bert_fgm_gain.png
```

错误分析：

```text
outputs/bert_freeze_classification_reports/
outputs/bert_freeze_confusion_matrices/
outputs/bert_freeze_predictions/
outputs/bert_freeze_wrong_cases/
```

---

## 15. Git 与大文件建议

建议不要提交以下大文件或自动生成文件：

```text
models/**/*.pth
models/**/*.pkl
models/**/tokenizer/
outputs/*.png
outputs/**/*.png
data/raw/
data/processed/
__pycache__/
```

建议保留：

```text
src/
models/bert/bert_unfreeze_last_n_layers.py
models/lstm/lstm_model.py
parameters/*/config.json
parameters/*/metrics.json
README.md
requirements.txt
main.py
```

---

## 16. 项目特点

- 从传统机器学习、RNN 到 BERT，覆盖不同复杂度的中文文本分类方案。
- 所有模型共用同一份 processed CSV 和统一 label map，便于公平比较。
- BERT 支持 full、frozen、partial 三类微调策略。
- freeze sweep 自动记录 trainable ratio 和 trainable parameters，便于分析性能与训练成本的关系。
- Partial-4/8 + Embedding Unfrozen + FGM 提供成对对比实验，能直接观察对抗训练对测试集表现的影响。
- 结果文件按 `models/`、`parameters/`、`outputs/` 分层保存，便于复现和分析。
