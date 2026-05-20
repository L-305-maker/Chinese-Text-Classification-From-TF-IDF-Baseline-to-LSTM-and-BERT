# 项目代码运行操作指南

本文档说明本项目从环境准备、数据处理、模型训练、模型评估到可视化汇总的完整运行方式。内容以当前代码为准，主要入口来自 `main.py` 和 `src/` 下各训练、评估、可视化模块。

所有命令建议在项目根目录执行：

```powershell
cd "D:\python\Chinese Text Classification From TF-IDF Baseline to LSTM and BERT"
```

## 1. 项目运行入口

项目统一入口是：

```powershell
python main.py --mode <任务名> [任务参数]
```

`main.py` 会根据 `--mode` 把命令转发给对应模块。

| `--mode` | 对应代码 | 作用 |
| --- | --- | --- |
| `data_processor` | `src.data_processor.process_raw_to_csv()` | 将 `data/raw/` 下的 CNews 原始 txt 文件转成 CSV |
| `LR` | `src.train_lr.main()` | 训练或检查 `TF-IDF + Logistic Regression` baseline |
| `Log_TF_IDF` | `src.train_lr.main()` | `LR` 的别名，功能完全相同 |
| `LSTM` | `src.train_lstm.main()` | 训练 LSTM 文本分类模型 |
| `BERT` | `src.train_bert.main()` | 训练 BERT、做冻结策略实验、评估已有 checkpoint、做温度校准 |
| `visualize` | `src.visualize.main()` | 根据 `configs/` 中的指标生成汇总文件和图表 |

查看总入口帮助：

```powershell
python main.py --help
```

查看某个子任务的参数：

```powershell
python main.py --mode BERT --help
python main.py --mode visualize --help
```

## 2. 环境准备

### 2.1 安装依赖

```powershell
pip install -r requirements.txt
```

`requirements.txt` 中的依赖及用途如下。

| 依赖 | 项目中的用途 |
| --- | --- |
| `jieba` | LR 和 LSTM 的中文分词 |
| `joblib` | 保存 sklearn 模型 |
| `matplotlib` | 绘制混淆矩阵、模型对比图、校准图 |
| `numpy` | 数值计算 |
| `pandas` | 读取和保存 CSV、汇总指标 |
| `scikit-learn` | TF-IDF、Logistic Regression、GridSearchCV、classification report |
| `torch` | LSTM 和 BERT 的训练、评估、checkpoint 保存 |
| `transformers` | 加载 `bert-base-chinese` tokenizer 和 BERT 模型 |

BERT 使用的预训练模型名称写在代码中：

```text
bert-base-chinese
```

首次运行 BERT 时，`transformers` 需要能读取本机缓存，或者能联网下载该模型。

### 2.2 目录约定

路径由 `src/utils/paths.py` 统一管理。

| 目录 | 作用 |
| --- | --- |
| `data/raw/` | 原始 CNews txt 数据 |
| `data/processed/` | 预处理后的 CSV 数据 |
| `configs/` | 每个实验的配置、指标、训练历史、标签映射 |
| `checkpoints/` | 模型权重、LR 模型文件、BERT tokenizer |
| `outputs/` | 分类报告、预测结果、错例、图表和校准结果 |

## 3. 数据处理

### 3.1 原始数据格式

项目读取以下原始文件：

```text
data/raw/cnews.train.txt
data/raw/cnews.val.txt
data/raw/cnews.test.txt
```

每行格式必须是：

```text
标签<TAB>正文
```

代码中支持的标签来自 `src/model_utils.py`：

```text
体育、财经、娱乐、家居、房产、教育、时尚、时政、游戏、科技
```

### 3.2 生成 processed CSV

```powershell
python main.py --mode data_processor
```

这个命令没有额外参数。

作用：

| 输入 | 输出 |
| --- | --- |
| `data/raw/cnews.train.txt` | `data/processed/train_data.csv` |
| `data/raw/cnews.val.txt` | `data/processed/val_data.csv` |
| `data/raw/cnews.test.txt` | `data/processed/test_data.csv` |

处理规则：

- 空行会跳过。
- 不满足 `标签<TAB>正文` 格式的行会跳过。
- 输出 CSV 包含 `text` 和 `label` 两列。
- 输出编码为 `utf-8-sig`，便于表格软件打开中文。
- 运行时会打印每个 split 的前几行和标签分布。

### 3.3 检查 processed 数据

```powershell
python main.py --mode LR --check-data
```

作用：

- 检查 `train_data.csv`、`val_data.csv`、`test_data.csv` 是否存在。
- 检查是否包含 `text`、`label` 两列。
- 检查 label 是否都属于项目支持的 10 个类别。
- 打印 train、val、test 的样本数量。

该命令只检查数据，不训练模型。

## 4. 推荐运行流程

### 4.1 从零开始完整运行

```powershell
pip install -r requirements.txt
python main.py --mode data_processor
python main.py --mode LR --check-data
python main.py --mode LR --small-grid
python main.py --mode LSTM
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8
python main.py --mode visualize --task both
```

### 4.2 快速验证代码链路

```powershell
python main.py --mode data_processor
python main.py --mode LR --check-data
python main.py --mode LR --sample-size 1000 --small-grid
python main.py --mode LSTM --epochs 1
```

BERT 训练耗时较长，建议先确认数据和 LR/LSTM 链路能跑通，再执行 BERT 训练。

## 5. LR / TF-IDF Baseline

命令格式：

```powershell
python main.py --mode LR [参数]
```

等价写法：

```powershell
python main.py --mode Log_TF_IDF [参数]
```

### 5.1 参数说明

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--sample-size` | `int` | `None` | 从训练集抽样指定数量样本训练，适合快速调试；代码会尽量按类别均衡抽样 |
| `--small-grid` | flag | 关闭 | 使用小搜索网格，只跑一组参数，速度更快 |
| `--check-data` | flag | 关闭 | 只检查 processed 数据，不训练模型 |

### 5.2 训练逻辑

代码位置：

```text
src/train_lr.py
```

运行流程：

1. 调用 `data_processor()` 读取 `data/processed/*.csv`。
2. 使用 `jieba.lcut()` 对中文文本分词。
3. 构建 sklearn `Pipeline`：`TfidfVectorizer` + `LogisticRegression`。
4. 使用 `GridSearchCV` 搜索参数。
5. 在测试集计算 `test_accuracy` 和 `test_macro_f1`。
6. 保存模型、最佳参数和指标。

默认模型配置：

| 组件 | 关键配置 |
| --- | --- |
| `TfidfVectorizer` | `max_features=5000`、`min_df=2`、`ngram_range=(1, 2)` |
| `LogisticRegression` | `solver="saga"`、`max_iter=10000`、`random_state=42` |
| `GridSearchCV` | 默认最多 5 折；如果样本少，会根据最少类别样本数自动减少折数 |

完整搜索网格：

| 参数 | 候选值 |
| --- | --- |
| `tfidf__max_features` | `5000`、`10000` |
| `tfidf__ngram_range` | `(1, 2)`、`(1, 3)` |
| `lr__C` | `0.01`、`0.1`、`1`、`10`、`100` |
| `lr__max_iter` | `10000`、`50000` |

`--small-grid` 搜索网格：

| 参数 | 候选值 |
| --- | --- |
| `tfidf__max_features` | `5000` |
| `tfidf__ngram_range` | `(1, 2)` |
| `lr__C` | `1` |
| `lr__max_iter` | `10000` |

### 5.3 常用命令

| 目标 | 命令 |
| --- | --- |
| 检查数据 | `python main.py --mode LR --check-data` |
| 快速训练 | `python main.py --mode LR --small-grid` |
| 抽样 1000 条快速训练 | `python main.py --mode LR --sample-size 1000 --small-grid` |
| 完整网格训练 | `python main.py --mode LR` |

### 5.4 输出产物

| 文件 | 内容 |
| --- | --- |
| `checkpoints/lr_tfidf/model.pkl` | 保存后的 sklearn pipeline |
| `configs/lr_tfidf/config.json` | 训练配置和参数网格 |
| `configs/lr_tfidf/best_params.json` | GridSearchCV 最佳参数 |
| `configs/lr_tfidf/metrics.json` | 测试集指标和交叉验证指标 |
| `configs/lr_tfidf/label_map.json` | 标签映射 |

## 6. LSTM 模型

命令格式：

```powershell
python main.py --mode LSTM [参数]
```

### 6.1 参数说明

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--embed_dim` | `int` | `64` | 词向量维度 |
| `--hidden_size` | `int` | `512` | LSTM hidden state 维度 |
| `--num_layers` | `int` | `1` | LSTM 层数 |
| `--dropout` | `float` | `0.3` | dropout 比例；当 `num_layers=1` 时，LSTM 内部 dropout 为 `0.0`，分类头前仍使用 dropout |
| `--batch_size` | `int` | `16` | DataLoader batch size |
| `--learning_rate` | `float` | `0.001` | Adam 优化器学习率 |
| `--epochs` | `int` | `5` | 训练轮数 |

### 6.2 训练逻辑

相关代码：

```text
src/train_lstm.py
src/dataset_lstm.py
models/lstm_classifier.py
```

运行流程：

1. 读取 `data/processed/*.csv`。
2. 使用 `jieba.cut()` 对训练文本分词。
3. 根据训练集构建词表，保留 `[PAD]=0`、`[UNK]=1`。
4. `collate_fn()` 会对每个 batch 动态 padding 到当前 batch 的最长文本长度。
5. 模型结构为 `Embedding -> LSTM -> Dropout -> Linear`。
6. 每个 epoch 训练后在验证集计算 loss、accuracy、macro F1。
7. 按验证集 macro F1 保存 `best_model.pth`。
8. 训练结束后加载最佳 checkpoint，在测试集评估。

### 6.3 常用命令

| 目标 | 命令 |
| --- | --- |
| 默认训练 | `python main.py --mode LSTM` |
| 只跑 1 个 epoch 调试 | `python main.py --mode LSTM --epochs 1` |
| 调整 batch size | `python main.py --mode LSTM --batch_size 32` |
| 调整词向量和 hidden 维度 | `python main.py --mode LSTM --embed_dim 128 --hidden_size 256` |
| 使用两层 LSTM | `python main.py --mode LSTM --num_layers 2 --dropout 0.3` |

### 6.4 输出产物

| 文件 | 内容 |
| --- | --- |
| `checkpoints/lstm/best_model.pth` | 验证集 macro F1 最优的 PyTorch checkpoint |
| `configs/lstm/config.json` | 模型结构和训练参数 |
| `configs/lstm/history.json` | 每个 epoch 的 train/val 指标 |
| `configs/lstm/metrics.json` | best val F1 和测试集指标 |
| `configs/lstm/vocab.json` | 训练集词表 |
| `configs/lstm/label_map.json` | 标签映射 |

## 7. BERT 模型

命令格式：

```powershell
python main.py --mode BERT [参数]
```

BERT 入口支持普通训练、冻结策略批量实验、FGM 对抗训练、已有 checkpoint 评估和温度校准。

### 7.1 基础训练参数

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--dropout` | `float` | `0.3` | BERT `[CLS]` 表示进入分类头前的 dropout |
| `--max_len` | `int` | `256` | tokenizer 最大序列长度，超出截断，不足 padding |
| `--batch_size` | `int` | `16` | 训练、验证、测试 batch size |
| `--epochs` | `int` | `5` | 训练轮数 |
| `--bert_lr` | `float` | `2e-5` | BERT backbone 参数学习率 |
| `--classifier_lr` | `float` | `1e-4` | 分类头参数学习率 |
| `--weight_decay` | `float` | `0.01` | AdamW weight decay |

### 7.2 微调策略参数

| 参数 | 类型/可选值 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--finetune_strategy` | `full`、`frozen`、`partial` | `full` | 控制 BERT 哪些参数参与训练 |
| `--unfreeze_last_n_layers` | `int` | `None` | 当策略为 `partial` 时，解冻最后 N 层 encoder；如果不传，代码会设为 `2` |

策略含义：

| 策略 | 训练参数 | 默认实验名 |
| --- | --- | --- |
| `full` | 全部 BERT 参数 + 分类头 | `bert_full_no_fgm` |
| `frozen` | 只训练分类头 | `bert_frozen_no_fgm` |
| `partial` | 解冻最后 N 层 encoder、pooler 和分类头 | `bert_partial_last_<N>_no_fgm` |

说明：

- 实验名由 `build_experiment_name()` 自动生成。
- 如果传入 `--experiment-name`，会覆盖自动实验名。
- checkpoint、config、output 都会以实验名作为目录名。

### 7.3 FGM 参数

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--use_fgm`、`--use-fgm` | flag | 关闭 | 启用 FGM 对抗训练 |
| `--fgm_epsilon`、`--fgm-epsilon` | `float` | `1.0` | FGM 扰动强度 |

FGM 使用限制：

- 必须使用 `--finetune_strategy partial`。
- `--unfreeze_last_n_layers` 只能是 `4` 或 `8`。
- 启用 FGM 时，代码会额外解冻 `bert.embeddings.*`。

合法示例：

```powershell
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4 --use-fgm
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8 --use-fgm
```

非法示例：

```powershell
python main.py --mode BERT --finetune_strategy full --use-fgm
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 2 --use-fgm
```

### 7.4 freeze sweep 参数

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--freeze-sweep` | flag | 关闭 | 批量运行 frozen、partial、full 多种微调策略 |
| `--partial_unfreeze_layers`、`--partial-unfreeze-layers` | `int` 列表 | `1 2 4 8 12` | sweep 中 partial 策略要测试的解冻层数 |
| `--freeze-output-dir` | `str` | `outputs/bert_freeze` | 保存 freeze summary 和图表的目录 |
| `--freeze-summary-filename` | `str` | `bert_freeze_summary.csv` | freeze summary 的文件名 |
| `--skip-freeze-visualize` | flag | 关闭 | 只保存 summary CSV，不自动画 freeze 图 |

sweep 规则：

- 默认运行 `frozen`、`partial last 1/2/4/8/12`、`full`。
- 如果同时加 `--use-fgm`，会对 partial last 4 和 partial last 8 增加 FGM 实验。
- 每个实验会单独保存到 `configs/<experiment>/`、`checkpoints/<experiment>/`、`outputs/<experiment>/`。

### 7.5 eval-only 与校准参数

| 参数 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--eval_only`、`--eval-only` | flag | 关闭 | 只加载已有 checkpoint 做评估，不重新训练 |
| `--experiment_name`、`--experiment-name` | `str` | `None` | 指定实验名，对应 `configs/<name>/` 和 `checkpoints/<name>/` |
| `--eval_batch_size`、`--eval-batch-size` | `int` | `None` | 只用于 `--eval-only`；不传则使用保存配置里的 batch size |
| `--calibrate` | flag | 关闭 | BERT 评估后执行 temperature scaling 校准 |
| `--calibration_bins`、`--calibration-bins` | `int` | `10` | ECE 和 reliability diagram 的分箱数量 |

`--eval-only` 依赖以下文件：

```text
checkpoints/<experiment>/best_model.pth
configs/<experiment>/config.json
```

开启 `--calibrate` 后，代码会：

1. 在验证集上收集 logits 并拟合 temperature。
2. 在测试集上计算校准前后的 NLL、ECE、accuracy。
3. 保存校准指标、校准分箱 CSV 和可靠性图。

### 7.6 BERT 常用命令

| 目标 | 命令 |
| --- | --- |
| 查看 BERT 帮助 | `python main.py --mode BERT --help` |
| full fine-tuning | `python main.py --mode BERT --finetune_strategy full` |
| frozen BERT | `python main.py --mode BERT --finetune_strategy frozen` |
| partial last 4 | `python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4` |
| partial last 8 | `python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8` |
| partial last 4 + FGM | `python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4 --use-fgm` |
| partial last 8 + FGM | `python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8 --use-fgm` |
| 指定实验名训练 | `python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8 --experiment-name bert_partial_last_8_trial_001` |
| 运行 freeze sweep | `python main.py --mode BERT --freeze-sweep` |
| freeze sweep 加 FGM | `python main.py --mode BERT --freeze-sweep --use-fgm` |
| 自定义 sweep 层数 | `python main.py --mode BERT --freeze-sweep --partial-unfreeze-layers 2 4 8` |
| 只评估已有模型 | `python main.py --mode BERT --eval-only --experiment-name bert_partial_last_8_no_fgm` |
| 评估时调整 batch | `python main.py --mode BERT --eval-only --experiment-name bert_partial_last_8_no_fgm --eval-batch-size 32` |
| 评估并做温度校准 | `python main.py --mode BERT --eval-only --experiment-name bert_partial_last_4_embedding_fgm --eval-batch-size 32 --calibrate` |
| 调整校准分箱 | `python main.py --mode BERT --eval-only --experiment-name bert_partial_last_4_embedding_fgm --calibrate --calibration-bins 15` |

### 7.7 BERT 输出产物

普通训练或 `--eval-only` 的主要输出：

| 文件 | 内容 |
| --- | --- |
| `checkpoints/<experiment>/best_model.pth` | 最优 PyTorch checkpoint |
| `checkpoints/<experiment>/tokenizer/` | 保存后的 tokenizer |
| `configs/<experiment>/config.json` | 训练配置 |
| `configs/<experiment>/history.json` | 训练历史；`--eval-only` 不生成新的训练历史 |
| `configs/<experiment>/metrics.json` | 参数量、best val 指标、test 指标、可选校准指标 |
| `configs/<experiment>/label_map.json` | 标签映射 |
| `outputs/<experiment>/<experiment>_classification_report.txt` | 分类报告 |
| `outputs/<experiment>/<experiment>_confusion_matrix.png` | 混淆矩阵 |
| `outputs/<experiment>/<experiment>_predictions.csv` | 测试集预测结果 |
| `outputs/<experiment>/<experiment>_wrong_cases.csv` | 测试集错例 |

温度校准输出：

| 文件 | 内容 |
| --- | --- |
| `outputs/calibration/<experiment>_calibration_metrics.json` | temperature、ECE、NLL、accuracy |
| `outputs/calibration/<experiment>_calibration_bins.csv` | 每个置信度分箱的数量、准确率、置信度 |
| `outputs/calibration/<experiment>_reliability_diagram.png` | reliability diagram |

freeze sweep 输出：

| 文件 | 内容 |
| --- | --- |
| `outputs/bert_freeze/bert_freeze_summary.csv` | freeze sweep 汇总表 |
| `outputs/bert_freeze/bert_freeze_scores.png` | 不同冻结策略测试集分数对比 |
| `outputs/bert_freeze/bert_freeze_trainable_params.png` | 可训练参数量对比 |
| `outputs/bert_freeze/bert_freeze_generalization_gap.png` | 泛化差距图 |
| `outputs/bert_freeze/bert_freeze_efficiency.png` | 参数效率图 |

## 8. 可视化和结果汇总

命令格式：

```powershell
python main.py --mode visualize [参数]
```

### 8.1 参数说明

| 参数 | 类型/可选值 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--task` | `models`、`bert_freeze`、`fgm`、`both` | `models` | 选择可视化任务 |
| `--models` | 目录名列表 | `lr_tfidf lstm bert_partial_last_8_no_fgm` | 指定参与普通模型对比的 `configs/` 子目录 |
| `--parameters-dir` | `str` | `configs` | 指标文件根目录 |
| `--output-dir` | `str` | `outputs/comparison` | 普通模型对比图输出目录 |
| `--bert-freeze-summary` | `str` | `outputs/bert_freeze/bert_freeze_summary.csv` | BERT freeze summary CSV 路径 |
| `--comparison-fgm-dir` | `str` | `outputs/fgm_comparison` | FGM 对比输出目录 |

### 8.2 task 说明

| `--task` | 读取内容 | 输出内容 |
| --- | --- | --- |
| `models` | `configs/<model>/metrics.json` 和可选 `history.json` | 模型总体对比表和图 |
| `bert_freeze` | `outputs/bert_freeze/bert_freeze_summary.csv`；如果不存在，会尝试从 `configs/` 重建 | BERT 冻结策略对比图 |
| `fgm` | `configs/bert_partial_last_4_*`、`configs/bert_partial_last_8_*` | partial last 4/8 有无 FGM 对比 |
| `both` | 上面三类都读取 | 一次性刷新常用汇总图 |

### 8.3 常用命令

| 目标 | 命令 |
| --- | --- |
| 默认模型总览 | `python main.py --mode visualize --task models` |
| 指定模型对比 | `python main.py --mode visualize --task models --models lr_tfidf lstm bert_partial_last_8_embedding_fgm` |
| BERT freeze 图 | `python main.py --mode visualize --task bert_freeze` |
| 指定 freeze summary | `python main.py --mode visualize --task bert_freeze --bert-freeze-summary outputs\bert_freeze\bert_freeze_summary.csv` |
| FGM 对比图 | `python main.py --mode visualize --task fgm` |
| 一次性刷新常用图 | `python main.py --mode visualize --task both` |
| 输出到自定义目录 | `python main.py --mode visualize --task models --output-dir outputs\my_comparison` |

### 8.4 输出产物

普通模型对比：

```text
outputs/comparison/model_metrics_summary.csv
outputs/comparison/model_metrics_summary.json
outputs/comparison/model_test_scores.png
outputs/comparison/model_metric_heatmap.png
outputs/comparison/model_test_loss.png
outputs/comparison/model_generalization_gap.png
outputs/comparison/training_loss_curves.png
outputs/comparison/training_acc_curves.png
outputs/comparison/training_f1_curves.png
```

BERT freeze 对比：

```text
outputs/bert_freeze/bert_freeze_summary.csv
outputs/bert_freeze/bert_freeze_scores.png
outputs/bert_freeze/bert_freeze_trainable_params.png
outputs/bert_freeze/bert_freeze_generalization_gap.png
outputs/bert_freeze/bert_freeze_efficiency.png
```

FGM 对比：

```text
outputs/fgm_comparison/bert_fgm_source_rows.csv
outputs/fgm_comparison/bert_fgm_comparison.csv
outputs/fgm_comparison/bert_fgm_comparison.png
outputs/fgm_comparison/bert_fgm_gain.png
```

## 9. 工程化运行建议

### 9.1 固定项目根目录

所有命令都建议在项目根目录执行。虽然代码通过 `Path(__file__)` 推导项目根目录，但固定工作目录可以减少命令、日志和相对路径混乱。

### 9.2 使用实验名隔离 BERT 结果

LR 和 LSTM 的实验名固定：

```text
lr_tfidf
lstm
```

重复运行会覆盖原目录下的配置、指标和 checkpoint。

BERT 支持指定实验名：

```powershell
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8 --experiment-name bert_partial_last_8_trial_001
```

建议正式实验都加 `--experiment-name`，避免覆盖已有结果。

### 9.3 先验证数据，再跑深度模型

推荐顺序：

```powershell
python main.py --mode LR --check-data
python main.py --mode LR --sample-size 1000 --small-grid
python main.py --mode LSTM --epochs 1
```

数据、标签和基础链路都正常后，再运行 BERT。

### 9.4 CPU 和 GPU

LSTM 和 BERT 会自动选择设备：

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

没有 CUDA 时，BERT 训练会比较慢。已有 checkpoint 的场景下，优先用 `--eval-only` 补评估、补图表。

### 9.5 结果完整性检查

训练或评估完成后，可以运行：

```powershell
python main.py --mode visualize --task both
```

如果该命令正常生成图表，通常说明 `configs/` 中的 `metrics.json`、`history.json` 和目录结构基本可用。

## 10. 常见问题

### 10.1 忘记传 `--mode`

现象：

```text
error: the following arguments are required: --mode
```

处理：

```powershell
python main.py --help
```

从支持的 mode 中选择一个任务。

### 10.2 BERT eval-only 找不到 checkpoint

现象：

```text
BERT checkpoint not found: checkpoints/<experiment>/best_model.pth
```

处理：

- 检查 `--experiment-name` 是否写对。
- 检查 `checkpoints/<experiment>/best_model.pth` 是否存在。
- 如果没有 checkpoint，需要先训练该实验。

### 10.3 FGM 参数组合不合法

FGM 只支持：

```powershell
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4 --use-fgm
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8 --use-fgm
```

不要与 `full`、`frozen` 或 partial last 1/2/12 一起使用。

### 10.4 visualize 提示 metrics 缺失

现象：

```text
[WARNING] Metrics file not found
```

处理：

- 检查 `configs/<model>/metrics.json` 是否存在。
- 对缺失模型先运行训练或 BERT `--eval-only`。
- `--models` 参数传的是 `configs/` 下的目录名，不是图中的展示名。

### 10.5 CSV 中文乱码

项目输出 CSV 多数使用 `utf-8-sig`。如果表格软件仍然乱码，建议使用 VS Code、PyCharm 或其他明确支持 UTF-8 的工具打开。
