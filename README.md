# Chinese Text Classification: TF-IDF, LSTM and BERT

本项目面向中文新闻文本分类任务，覆盖从传统机器学习 baseline 到深度学习模型和预训练语言模型微调的完整实验流程。

当前支持：

- 数据预处理：将原始 `cnews.*.txt` 转换为统一 CSV。
- Baseline：TF-IDF + Logistic Regression。
- 深度学习模型：LSTM 文本分类器。
- 预训练模型：`bert-base-chinese` 文本分类器。
- BERT 微调策略对比：`frozen`、`partial`、`full`。
- FGM 对抗训练对比：partial 4/8 层下对比 no FGM 与 FGM。
- 可视化：训练曲线、模型对比、BERT 微调策略对比、FGM 影响对比。

## 1. 项目结构

```text
.
├── main.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_processor.py
│   ├── dataset_bert.py
│   ├── dataset_lstm.py
│   ├── train_lr.py
│   ├── train_bert.py
│   ├── train_lstm.py
│   ├── adversarial.py
│   ├── evaluate.py
│   ├── model_utils.py
│   ├── visualize.py
│   └── utils/
│       └── paths.py
├── models/
│   ├── bert_classifier.py
│   └── lstm_classifier.py
├── configs/
│   └── <model_or_experiment>/
│       ├── config.json
│       ├── history.json
│       ├── label_map.json
│       └── metrics.json
├── checkpoints/
│   └── <model_or_experiment>/
│       ├── best_model.pth
│       └── tokenizer/
└── outputs/
    ├── comparison/
    ├── fgm_comparison/
    ├── bert_freeze/
    └── <model_or_experiment>/
```

目录职责：

- `data/raw/`：原始数据。
- `data/processed/`：处理后的 CSV 数据。
- `src/`：训练、评估、数据处理、可视化等源码。
- `models/`：只存放模型结构定义，不存放训练权重。
- `configs/`：保存每个模型或实验的 `config`、`history`、`metrics`、`label_map` 等 JSON 数据。
- `checkpoints/`：保存模型权重、sklearn 模型文件、BERT tokenizer。
- `outputs/comparison/`：保存 LR、LSTM、BERT 三类模型的对比数据和图。
- `outputs/fgm_comparison/`：保存 FGM 与 no FGM 的对比数据和图。
- `outputs/bert_freeze/`：保存不同 BERT 微调策略的对比图。

更详细的结构说明见：

```text
docs/PROJECT_STRUCTURE.md
```

## 2. 环境安装

建议使用 Python 3.10 或以上版本。

```bash
conda create -n textcls python=3.10
conda activate textcls
pip install -r requirements.txt
```

依赖包括：

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

如果使用 GPU 训练 BERT，请根据本机 CUDA 版本安装匹配的 PyTorch。

## 3. 数据准备

原始数据放在：

```text
data/raw/
├── cnews.train.txt
├── cnews.val.txt
├── cnews.test.txt
└── cnews.vocab.txt
```

原始文本格式为：

```text
label<TAB>text
```

运行预处理：

```bash
python main.py --mode data_processor
```

生成结果：

```text
data/processed/
├── train_data.csv
├── val_data.csv
└── test_data.csv
```

CSV 包含两列：

```text
text,label
```

## 4. 统一入口

根目录的 `main.py` 是项目总入口。

```bash
python main.py --mode data_processor
python main.py --mode LR
python main.py --mode LSTM
python main.py --mode BERT
python main.py --mode visualize
```

兼容旧入口：

```bash
python main.py --mode Log_TF_IDF
```

等价于：

```bash
python main.py --mode LR
```

## 5. 训练 TF-IDF + Logistic Regression

检查数据是否可读取：

```bash
python main.py --mode LR --check-data
```

快速训练小网格：

```bash
python main.py --mode LR --small-grid
```

使用部分训练样本调试：

```bash
python main.py --mode LR --sample-size 5000 --small-grid
```

完整训练：

```bash
python main.py --mode LR
```

输出位置：

```text
configs/lr_tfidf/
checkpoints/lr_tfidf/
```

## 6. 训练 LSTM

默认训练：

```bash
python main.py --mode LSTM
```

指定常用参数：

```bash
python main.py --mode LSTM --epochs 5 --batch_size 16 --embed_dim 64 --hidden_size 512
```

可用参数：

```text
--embed_dim
--hidden_size
--num_layers
--dropout
--batch_size
--learning_rate
--epochs
```

输出位置：

```text
configs/lstm/
checkpoints/lstm/
```

## 7. 训练 BERT

默认训练 full fine-tuning：

```bash
python main.py --mode BERT
```

冻结 BERT，只训练分类头：

```bash
python main.py --mode BERT --finetune_strategy frozen
```

全量微调：

```bash
python main.py --mode BERT --finetune_strategy full
```

部分层微调，例如只解冻最后 4 层：

```bash
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4
```

常用参数：

```text
--dropout
--max_len
--batch_size
--epochs
--bert_lr
--classifier_lr
--weight_decay
--finetune_strategy
--unfreeze_last_n_layers
--experiment_name
```

输出位置：

```text
configs/<bert_experiment_name>/
checkpoints/<bert_experiment_name>/
outputs/<bert_experiment_name>/
```

## 8. BERT 微调策略实验

运行 frozen、partial、full 的 sweep：

```bash
python main.py --mode BERT --freeze-sweep
```

指定 partial 解冻层数：

```bash
python main.py --mode BERT --freeze-sweep --partial-unfreeze-layers 1 2 4 8 12
```

同时加入 partial 4/8 的 FGM 对比实验：

```bash
python main.py --mode BERT --freeze-sweep --use-fgm --partial-unfreeze-layers 1 2 4 8 12
```

FGM 参数：

```text
--use-fgm
--fgm-epsilon
```

说明：

- FGM 目前只支持 partial fine-tuning 且解冻层数为 4 或 8。
- 使用 FGM 时会同时解冻 embedding 层以支持对抗扰动。

## 9. 评估和可视化

生成 LR、LSTM、BERT 三类模型对比：

```bash
python main.py --mode visualize --task models
```

输出：

```text
outputs/comparison/
├── model_metrics_summary.csv
├── model_metrics_summary.json
├── model_test_scores.png
├── model_metric_heatmap.png
├── model_test_loss.png
├── model_generalization_gap.png
├── training_loss_curves.png
├── training_acc_curves.png
└── training_f1_curves.png
```

生成 BERT 不同微调策略对比：

```bash
python main.py --mode visualize --task bert_freeze
```

输出：

```text
outputs/bert_freeze/
├── bert_freeze_summary.csv
├── bert_freeze_scores.png
├── bert_freeze_trainable_params.png
├── bert_freeze_generalization_gap.png
└── bert_freeze_efficiency.png
```

生成 partial 4/8 的 FGM 对比：

```bash
python main.py --mode visualize --task fgm
```

输出：

```text
outputs/fgm_comparison/
├── bert_fgm_source_rows.csv
├── bert_fgm_comparison.csv
├── bert_fgm_comparison.png
├── bert_fgm_gain.png
└── bert_fgm_missing_runs.csv
```

一次性生成所有可视化：

```bash
python main.py --mode visualize --task both
```

## 10. 结果文件说明

`configs/<experiment>/config.json`：
记录实验配置，例如 batch size、学习率、微调策略、解冻层数等。

`configs/<experiment>/history.json`：
记录多轮训练过程中的 loss、accuracy、macro F1。

`configs/<experiment>/metrics.json`：
记录测试集指标、最佳验证集指标、参数量统计等。

`configs/<experiment>/label_map.json`：
记录标签到 ID 的映射关系。

`checkpoints/<experiment>/best_model.pth`：
PyTorch 模型权重。

`checkpoints/<experiment>/model.pkl`：
sklearn 模型文件，例如 TF-IDF + LR。

`outputs/<experiment>/`：
单个模型或单个实验的 report、confusion matrix、predictions、wrong cases。

## 11. 当前注意事项

当前已有数据中，partial 8 的 FGM 与 no FGM 对照是完整的。

partial 4 目前缺少 `bert_partial_last_4_embedding_fgm`，所以：

```text
outputs/fgm_comparison/bert_fgm_missing_runs.csv
```

会记录该缺失项。训练完 partial 4 + FGM 后，重新运行：

```bash
python main.py --mode visualize --task fgm
```

即可自动更新 FGM 对比图和 CSV。

## 12. Git 管理建议

建议提交：

```text
main.py
src/
models/
configs/README.md
docs/
requirements.txt
README.md
.gitignore
```

通常不建议提交：

```text
data/raw/
data/processed/
checkpoints/
outputs/
*.pth
*.pkl
```

这样可以保持仓库轻量，同时保留代码、配置说明和复现实验所需的结构。

