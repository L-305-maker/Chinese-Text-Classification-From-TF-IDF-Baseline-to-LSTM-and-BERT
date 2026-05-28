# 项目结构说明

本文档说明当前项目目录、数据流和主要模块职责。项目已经切换到 `online_shopping_10_cats` 中文电商评论十分类数据集。

## 目录概览

```text
|- main.py
|- requirements.txt
|- README.md
|- data/
|  |- raw/
|  |  |- online_shopping_10_cats/
|  |     |- online_shopping_10_cats.csv
|  |- processed/
|     |- train_data.csv
|     |- val_data.csv
|     |- test_data.csv
|
|- models/
|  |- bert_classifier.py
|  |- lstm_classifier.py
|
|- src/
|  |- adversarial.py
|  |- calibration.py
|  |- data_processor.py
|  |- dataset_bert.py
|  |- dataset_lstm.py
|  |- evaluate.py
|  |- model_utils.py
|  |- train_bert.py
|  |- train_lr.py
|  |- train_lstm.py
|  |- training_utils.py
|  |- visualize.py
|  |- utils/
|     |- paths.py
|
|- configs/
|- checkpoints/
|- outputs/
|- docs/
```

## 数据流

```text
data/raw/online_shopping_10_cats/online_shopping_10_cats.csv
  -> src/data_processor.py
  -> data/processed/train_data.csv
  -> data/processed/val_data.csv
  -> data/processed/test_data.csv
  -> src/train_lr.py / src/train_lstm.py / src/train_bert.py
  -> configs/<experiment>/
  -> checkpoints/<experiment>/
  -> outputs/<experiment>/
```

原始 CSV 的 `cat` 列会映射为项目的 `label`，`review` 列会映射为项目的 `text`。原始情感 `label` 列不参与当前十分类任务。

## 当前类别

`src/model_utils.py` 中的 `LABEL2ID` 是全项目唯一标签来源：

```text
书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店
```

LR、LSTM 和 BERT 都通过 `src.data_processor.build_id_map()` 使用同一套标签映射。

## 入口调度

`main.py` 是统一命令入口：

| mode | 调度目标 |
| --- | --- |
| `data_processor` | `src.data_processor.process_raw_to_csv` |
| `LR` / `Log_TF_IDF` | `src.train_lr.main` |
| `LSTM` | `src.train_lstm.main` |
| `BERT` | `src.train_bert.main` |
| `visualize` | `src.visualize.main` |

## 源码职责

| 文件 | 职责 |
| --- | --- |
| `src/data_processor.py` | 读取 `online_shopping_10_cats` 原始 CSV、生成 processed CSV、校验标签、执行标签映射 |
| `src/dataset_lstm.py` | LSTM 分词、词表、编码、padding 和 DataLoader |
| `src/dataset_bert.py` | BERT tokenizer、Dataset 和 DataLoader |
| `src/train_lr.py` | TF-IDF + Logistic Regression 训练、验证集调参和保存 |
| `src/train_lstm.py` | LSTM 训练、验证、测试和实验产物保存 |
| `src/train_bert.py` | BERT full/frozen/partial、FGM、eval-only、校准和 sweep 调度 |
| `src/model_utils.py` | 标签映射、实验初始化、配置/指标/checkpoint 保存、optimizer 构造 |
| `src/training_utils.py` | epoch 级 loss、accuracy、macro F1 统计 |
| `src/evaluate.py` | 分类报告、混淆矩阵、预测结果和错误样本输出 |
| `src/visualize.py` | 模型对比、BERT freeze summary、FGM 对比和图表生成 |
| `src/utils/paths.py` | 项目路径常量和目录创建 |
| `models/bert_classifier.py` | BERT 分类器和冻结策略 |
| `models/lstm_classifier.py` | LSTM 分类器 |

## 关键产物

| 路径 | 内容 |
| --- | --- |
| `data/processed/*.csv` | 当前数据集的统一 `text,label` split |
| `configs/<experiment>/config.json` | 训练参数和模型配置 |
| `configs/<experiment>/label_map.json` | 当前标签映射 |
| `configs/<experiment>/metrics.json` | 训练或评估指标 |
| `configs/lstm/vocab.json` | LSTM 训练词表 |
| `checkpoints/<experiment>/best_model.pth` | PyTorch 最优 checkpoint |
| `checkpoints/lr_tfidf/model.pkl` | sklearn pipeline |
| `outputs/<experiment>/` | 分类报告、混淆矩阵、预测结果和错误样本 |
| `outputs/comparison/` | 模型总览图和 summary |
| `outputs/bert_freeze/` | BERT 冻结策略对比 |
| `outputs/fgm_comparison/` | partial 4/8 的 FGM 对比 |
| `outputs/calibration/` | BERT 温度校准结果 |

## 维护约定

切换数据集后，旧 checkpoint、指标和图表不再代表当前数据集表现。正式实验应按以下顺序重新生成：

```powershell
python main.py --mode data_processor
python main.py --mode LR --check-data
python main.py --mode LR --small-grid
python main.py --mode LSTM
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8
python main.py --mode visualize --task both
```

如果只做代码链路检查，可以使用小样本和单 epoch：

```powershell
python main.py --mode LR --sample-size 1000 --small-grid
python main.py --mode LSTM --epochs 1 --sample-size 1000
```
