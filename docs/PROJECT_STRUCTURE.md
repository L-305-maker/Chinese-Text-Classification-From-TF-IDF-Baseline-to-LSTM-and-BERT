# 项目结构说明

当前项目按“数据、源码、模型定义、实验配置、输出结果”拆分：

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

## 目录职责

- `data/raw/`：原始数据，例如 `cnews.train.txt`。
- `data/processed/`：预处理后的数据，例如 `train_data.csv`。
- `src/data_processor.py`：原始数据读取、清洗、CSV 生成和标签映射。
- `src/dataset_lstm.py`：LSTM 输入数据、词表、编码、DataLoader。
- `src/dataset_bert.py`：BERT tokenizer、Dataset、DataLoader。
- `src/train_lr.py`：TF-IDF + Logistic Regression baseline 的训练逻辑。
- `src/train_lstm.py`：LSTM 深度学习模型的训练逻辑。
- `src/train_bert.py`：BERT 预训练模型、冻结策略、FGM 训练调度。
- `src/adversarial.py`：FGM 对抗训练实现。
- `src/evaluate.py`：通用评估、classification report、混淆矩阵、预测结果、错例分析。
- `src/model_utils.py`：通用保存、路径、参数统计、optimizer 构造等工具函数。
- `src/visualize.py`：训练曲线、三模型对比、BERT 微调策略对比、FGM 对比可视化。
- `models/`：只放模型结构定义。
- `configs/`：每个模型或实验的 `label_map`、`history`、`config`、`metrics` 等数据。
- `checkpoints/`：模型权重、sklearn pickle、BERT tokenizer。这个目录是训练产物，不放源码。
- `outputs/comparison/`：TF-IDF + LR、LSTM、BERT 三类模型的对比数据和图。
- `outputs/fgm_comparison/`：partial 4/8 下 FGM 与 no FGM 的对比数据和图。
- `outputs/bert_freeze/`：不同 BERT 微调策略的性能变化图。

## 常用命令

```bash
python main.py --mode data_processor
python main.py --mode LR
python main.py --mode LSTM
python main.py --mode BERT
python main.py --mode visualize --task models
python main.py --mode visualize --task bert_freeze
python main.py --mode visualize --task fgm
python main.py --mode visualize --task both
```

`--task models` 会写入 `outputs/comparison/`。

`--task bert_freeze` 会写入 `outputs/bert_freeze/`。

`--task fgm` 会写入 `outputs/fgm_comparison/`。

