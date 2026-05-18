# Chinese Text Classification: TF-IDF, LSTM and BERT

这个项目用于中文新闻文本分类实验，覆盖从传统机器学习 baseline 到深度学习模型、预训练模型微调和可视化分析的完整流程。

当前支持：

- 数据预处理：将 `cnews.*.txt` 转换为统一的 CSV 文件。
- Baseline：TF-IDF + Logistic Regression。
- 深度学习模型：基于 `jieba` 分词和 PyTorch 的 LSTM 分类器。
- 预训练模型：基于 `bert-base-chinese` 的 BERT 文本分类器。
- BERT 微调策略对比：`full`、`frozen`、`partial`。
- BERT FGM 对抗训练：用于 partial 4/8 层解冻实验。
- 实验结果保存：配置、训练历史、指标、模型权重和 tokenizer。
- 可视化：模型对比、训练曲线、BERT 冻结策略对比、FGM 对比。

## 项目结构

```text
.
├── main.py
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── cnews.train.txt
│   │   ├── cnews.val.txt
│   │   └── cnews.test.txt
│   └── processed/
│       ├── train_data.csv
│       ├── val_data.csv
│       └── test_data.csv
├── models/
│   ├── bert_classifier.py
│   └── lstm_classifier.py
├── src/
│   ├── adversarial.py
│   ├── data_processor.py
│   ├── dataset_bert.py
│   ├── dataset_lstm.py
│   ├── evaluate.py
│   ├── model_utils.py
│   ├── train_bert.py
│   ├── train_lr.py
│   ├── train_lstm.py
│   ├── training_utils.py
│   ├── visualize.py
│   └── utils/
│       └── paths.py
├── configs/
├── checkpoints/
├── outputs/
└── docs/
```

核心目录说明：

- `data/raw/`：原始 CNews 文本数据，格式为 `label<TAB>text`。
- `data/processed/`：预处理后的 CSV 数据，包含 `text` 和 `label` 两列。
- `models/`：只存放模型结构定义。
- `src/`：训练、评估、数据加载、可视化和公共工具代码。
- `configs/`：保存每次实验的 `config.json`、`history.json`、`metrics.json`、`label_map.json`。
- `checkpoints/`：保存模型权重、sklearn 模型和 BERT tokenizer。
- `outputs/`：保存分类报告、混淆矩阵、预测结果、错误样本和对比图。

## 环境安装

建议使用 Python 3.10 或更高版本。

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
```

如果使用 GPU 训练 BERT，请根据本机 CUDA 版本安装匹配的 PyTorch。

## 数据准备

将原始数据放到 `data/raw/`：

```text
data/raw/
├── cnews.train.txt
├── cnews.val.txt
└── cnews.test.txt
```

原始文本格式：

```text
label<TAB>text
```

执行预处理：

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

当前标签映射定义在 `src/model_utils.py`，包含：

```text
体育、财经、娱乐、家居、房产、教育、时尚、时政、游戏、科技
```

## 统一入口

所有主要任务都可以通过根目录的 `main.py` 运行：

```bash
python main.py --mode data_processor
python main.py --mode LR
python main.py --mode LSTM
python main.py --mode BERT
python main.py --mode visualize
```

兼容入口：

```bash
python main.py --mode Log_TF_IDF
```

等价于：

```bash
python main.py --mode LR
```

## 训练 TF-IDF + Logistic Regression

检查 processed 数据是否可读、标签是否合法：

```bash
python main.py --mode LR --check-data
```

快速小网格训练：

```bash
python main.py --mode LR --small-grid
```

使用部分样本调试：

```bash
python main.py --mode LR --sample-size 5000 --small-grid
```

完整训练：

```bash
python main.py --mode LR
```

常用参数：

```text
--check-data
--sample-size
--small-grid
```


## 训练 LSTM

默认训练：

```bash
python main.py --mode LSTM
```

指定参数：

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


## 训练 BERT

默认训练使用 full fine-tuning：

```bash
python main.py --mode BERT
```

冻结 BERT，只训练分类头：

```bash
python main.py --mode BERT --finetune_strategy frozen
```

部分解冻最后 4 层：

```bash
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4
```

自定义学习率：

```bash
python main.py --mode BERT --bert_lr 2e-5 --classifier_lr 1e-4 --weight_decay 0.01
```

常用参数：

```text
--dropout
--max_len
--batch_size
--epochs
--finetune_strategy
--unfreeze_last_n_layers
--bert_lr
--classifier_lr
--weight_decay
--experiment_name
```

BERT 默认实验名由策略自动生成，例如：

```text
bert_full_no_fgm
bert_partial_last_4_no_fgm
bert_partial_last_4_embedding_fgm
```

也可以通过 `--experiment_name` 覆盖。


## BERT 冻结策略对比

运行 frozen、partial 和 full 的批量对比实验：

```bash
python main.py --mode BERT --freeze-sweep
```

指定 partial 解冻层数：

```bash
python main.py --mode BERT --freeze-sweep --partial-unfreeze-layers 1 2 4 8 12
```

跳过自动可视化：

```bash
python main.py --mode BERT --freeze-sweep --skip-freeze-visualize
```

## BERT FGM 对抗训练

FGM 当前只支持 partial fine-tuning 且解冻层数为 4 或 8。启用 FGM 时，代码会同时解冻 BERT embedding 参数以进行扰动。

示例：

```bash
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4 --use-fgm
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8 --use-fgm
```

指定扰动强度：

```bash
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4 --use-fgm --fgm-epsilon 1.0
```

在 freeze sweep 中加入 FGM 对比：

```bash
python main.py --mode BERT --freeze-sweep --use-fgm
```

## 仅评估已保存的 BERT 实验

如果已经存在 checkpoint，可以不重新训练，直接导出评估产物：

```bash
python main.py --mode BERT --eval-only --experiment-name bert_partial_last_4_no_fgm
```

可选指定评估 batch size：

```bash
python main.py --mode BERT --eval-only --experiment-name bert_partial_last_4_no_fgm --eval-batch-size 32
```

## 可视化

默认对比 LR、LSTM 和 BERT：

```bash
python main.py --mode visualize
```

只生成模型对比图：

```bash
python main.py --mode visualize --task models
```

只生成 BERT 冻结策略图：

```bash
python main.py --mode visualize --task bert_freeze
```

只生成 FGM 对比图：

```bash
python main.py --mode visualize --task fgm
```

同时生成模型对比、BERT 冻结策略图和 FGM 对比图：

```bash
python main.py --mode visualize --task both
```

指定参与对比的模型目录：

```bash
python main.py --mode visualize --task models --models lr_tfidf lstm bert_partial_last_8_no_fgm
```

## 模块职责

```text
main.py
```

统一命令入口，按 `--mode` 将任务分发到具体模块。

```text
src/data_processor.py
```

负责原始数据转 CSV、读取 processed 数据、标签转 ID。

```text
src/dataset_lstm.py
```

负责 LSTM 的分词、词表构建、编码、padding 和 DataLoader。

```text
src/dataset_bert.py
```

负责 BERT tokenizer、BERT Dataset 和 DataLoader。

```text
models/lstm_classifier.py
models/bert_classifier.py
```

模型结构定义。

```text
src/train_lr.py
src/train_lstm.py
src/train_bert.py
```

分别负责 TF-IDF + LR、LSTM、BERT 的训练流程。

```text
src/training_utils.py
```

训练/验证 epoch 的公共指标统计工具。

```text
src/model_utils.py
```

实验初始化、配置保存、指标保存、模型保存、参数统计和优化器构建。

```text
src/evaluate.py
```

分类指标、分类报告、混淆矩阵、预测结果和错误样本保存。

```text
src/visualize.py
```

模型指标汇总、训练曲线、BERT 冻结策略对比和 FGM 对比可视化。

```text
src/utils/paths.py
```

统一维护项目路径和目录创建函数。

## 推荐实验流程

1. 准备原始数据到 `data/raw/`。
2. 运行 `python main.py --mode data_processor`。
3. 运行 `python main.py --mode LR --check-data` 检查数据。
4. 先用 `python main.py --mode LR --small-grid` 得到 baseline。
5. 训练 LSTM：`python main.py --mode LSTM`。
6. 训练 BERT：`python main.py --mode BERT`。
7. 做 BERT 策略对比：`python main.py --mode BERT --freeze-sweep`。
8. 生成图表：`python main.py --mode visualize --task both`。

## 快速自检

检查 Python 文件是否能通过编译：

```bash
python -m compileall -q main.py src models
```

查看统一入口帮助：

```bash
python main.py --help
```

查看某个子任务帮助，例如：

```bash
python main.py --mode BERT --help
python main.py --mode visualize --help
```

## 说明

`.gitignore` 默认忽略数据、checkpoint、虚拟环境、缓存和部分实验产物。训练产生的大文件建议保留在本地，不建议直接提交到版本库。
