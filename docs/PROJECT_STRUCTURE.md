# 项目结构说明

本文档说明当前代码、实验产物和图谱输出之间的关系。项目按“数据 -> 模型 -> 训练/评估/校准 -> 可视化 -> 文档”组织。

## 目录总览

```text
|- main.py
|- requirements.txt
|- README.md
|- data/
|  |- raw/
|  |- processed/
|
|- models/
|  |- __init__.py
|  |- bert_classifier.py
|  |- lstm_classifier.py
|
|- src/
|  |- __init__.py
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
|  |- <model_or_experiment>/
|     |- config.json
|     |- history.json
|     |- label_map.json
|     |- metrics.json
|
|- checkpoints/
|  |- <model_or_experiment>/
|     |- best_model.pth
|     |- tokenizer/
|
|- outputs/
|  |- comparison/
|  |- bert_freeze/
|  |- fgm_comparison/
|  |- calibration/
|  |- <model_or_experiment>/
|
|- docs/
   |- PROJECT_STRUCTURE.md
```

## 入口与调度

`main.py` 是统一命令入口，通过 `--mode` 分发到具体模块：

- `data_processor` -> `src.data_processor.process_raw_to_csv`
- `LR` / `Log_TF_IDF` -> `src.train_lr.main`
- `LSTM` -> `src.train_lstm.main`
- `BERT` -> `src.train_bert.main`
- `visualize` -> `src.visualize.main`

常用命令：

```bash
python main.py --mode data_processor
python main.py --mode LR
python main.py --mode LSTM
python main.py --mode BERT
python main.py --mode visualize --task both
```

## 源码职责

| 文件 | 职责 |
| --- | --- |
| `src/data_processor.py` | 读取 CNews 原始文本、生成 processed CSV、执行标签映射。 |
| `src/dataset_lstm.py` | LSTM 文本分词、词表、编码、padding 和 DataLoader。 |
| `src/dataset_bert.py` | BERT tokenizer、Dataset 和 DataLoader。 |
| `src/train_lr.py` | TF-IDF + Logistic Regression baseline 训练、调参和保存。 |
| `src/train_lstm.py` | LSTM 训练、验证、测试和实验产物保存。 |
| `src/train_bert.py` | BERT full/frozen/partial、FGM、eval-only、校准接入和 sweep 调度。 |
| `src/adversarial.py` | FGM 对抗扰动、参数备份和恢复。 |
| `src/calibration.py` | 收集 logits、拟合 temperature、计算 ECE/NLL、输出可靠性图。 |
| `src/evaluate.py` | 分类报告、混淆矩阵、预测结果、错例分析。 |
| `src/model_utils.py` | 实验初始化、配置/指标保存、checkpoint 保存、参数统计、optimizer 构造。 |
| `src/training_utils.py` | epoch 级 loss、accuracy、macro F1 统计。 |
| `src/visualize.py` | 三模型对比、BERT freeze summary、FGM 对比和图谱刷新。 |
| `src/utils/paths.py` | 项目路径常量和目录创建函数。 |
| `models/bert_classifier.py` | BERT 分类器结构和冻结策略实现。 |
| `models/lstm_classifier.py` | LSTM 分类器结构。 |

## 实验命名

BERT 默认按策略生成实验名：

```text
bert_full_no_fgm
bert_frozen
bert_partial_last_4_no_fgm
bert_partial_last_4_embedding_fgm
bert_partial_last_8_no_fgm
bert_partial_last_8_embedding_fgm
```

也可通过 `--experiment-name` 覆盖。

FGM 当前限制在 partial last 4/8。启用 FGM 时，代码会额外解冻 BERT embedding，以便对 embedding 参数施加扰动。

## 数据和产物流向

```text
data/raw/*.txt
  -> src/data_processor.py
  -> data/processed/*.csv
  -> train_lr.py / train_lstm.py / train_bert.py
  -> configs/<experiment>/
  -> checkpoints/<experiment>/
  -> outputs/<experiment>/
  -> src/visualize.py
  -> outputs/comparison, outputs/bert_freeze, outputs/fgm_comparison
```

BERT `--calibrate` 额外生成：

```text
train_bert.py eval result
  -> src/calibration.py
  -> outputs/calibration/<experiment>_calibration_metrics.json
  -> outputs/calibration/<experiment>_calibration_bins.csv
  -> outputs/calibration/<experiment>_reliability_diagram.png
  -> configs/<experiment>/metrics.json
```

## 关键输出

| 路径 | 内容 |
| --- | --- |
| `configs/<experiment>/config.json` | 训练参数、模型策略、FGM/校准配置。 |
| `configs/<experiment>/history.json` | 每个 epoch 的 train/val loss、accuracy、macro F1。 |
| `configs/<experiment>/metrics.json` | 最终 test 指标、参数统计、可选校准指标。 |
| `checkpoints/<experiment>/best_model.pth` | 最优模型权重。 |
| `outputs/<experiment>/*classification_report.txt` | 测试集分类报告。 |
| `outputs/<experiment>/*confusion_matrix.png` | 混淆矩阵。 |
| `outputs/<experiment>/*predictions.csv` | 测试集预测明细。 |
| `outputs/<experiment>/*wrong_cases.csv` | 错误样本。 |
| `outputs/comparison/` | LR、LSTM、BERT 总览图和 summary。 |
| `outputs/bert_freeze/` | BERT 冻结策略 summary 和图。 |
| `outputs/fgm_comparison/` | partial 4/8 FGM 对比 CSV 和图。 |
| `outputs/calibration/` | BERT 温度校准指标、分箱和可靠性图。 |

## 当前补齐项

`bert_partial_last_4_embedding_fgm` 已通过 `eval-only --calibrate` 补齐：

- `configs/bert_partial_last_4_embedding_fgm/metrics.json`
- `outputs/bert_partial_last_4_embedding_fgm/bert_partial_last_4_embedding_fgm_classification_report.txt`
- `outputs/bert_partial_last_4_embedding_fgm/bert_partial_last_4_embedding_fgm_confusion_matrix.png`
- `outputs/bert_partial_last_4_embedding_fgm/bert_partial_last_4_embedding_fgm_predictions.csv`
- `outputs/bert_partial_last_4_embedding_fgm/bert_partial_last_4_embedding_fgm_wrong_cases.csv`
- `outputs/calibration/bert_partial_last_4_embedding_fgm_*`

刷新后的 FGM 对比：

- partial last 4：FGM test macro F1 `0.970003`，较 no FGM 提升 `0.003315`。
- partial last 8：FGM test macro F1 `0.971554`，较 no FGM 提升 `0.001897`。
- `outputs/fgm_comparison/bert_fgm_missing_runs.csv` 已消失，表示 partial 4/8 的 with/without FGM 组合都完整。

## 推荐维护顺序

1. 修改模型或训练代码后运行 `python -m compileall -q main.py src models`。
2. 对已有 BERT checkpoint 使用 `--eval-only` 补评估，不要无意重训。
3. 需要校准时追加 `--calibrate`，并优先设置合适的 `--eval-batch-size`。
4. 评估补齐后运行 `python main.py --mode visualize --task fgm` 刷新 FGM 图。
5. 若新增实验需要进 freeze summary，用不存在的 `--bert-freeze-summary` 路径触发从 `configs/` 重建。

