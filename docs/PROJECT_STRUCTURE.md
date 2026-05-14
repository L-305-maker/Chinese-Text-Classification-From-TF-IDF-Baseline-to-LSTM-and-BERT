# 项目结构说明

本项目已经把“源码”和“训练产物”拆开管理，避免模型定义、checkpoint、指标文件和可视化报告混在同一个目录。

```text
.
├── main.py
├── requirements.txt
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── notebooks/
├── reports/
│   ├── bert_freeze/
│   ├── figures/
│   ├── model_comparison/
│   └── tables/
├── comparison_fgm/
│   ├── bert_fgm_comparison.csv
│   ├── bert_fgm_comparison.png
│   ├── bert_fgm_gain.png
│   ├── bert_fgm_missing_runs.csv
│   └── bert_fgm_source_rows.csv
├── runs/
│   └── <experiment_name>/
│       ├── best_model.pth
│       ├── config.json
│       ├── history.json
│       ├── label_map.json
│       ├── metrics.json
│       ├── reports/
│       └── tokenizer/
└── src/
    ├── data_process.py
    ├── datasets_bert.py
    ├── datasets_lstm.py
    ├── evaluate.py
    ├── lr_train.py
    ├── model_utils.py
    ├── models/
    │   ├── bert_classifier.py
    │   └── lstm_classifier.py
    ├── train_bert.py
    ├── train_lstm.py
    ├── utils/
    │   └── paths.py
    └── visualize.py
```

## 目录职责

- `src/`：项目源码。
- `src/models/`：模型结构定义，只放代码，不放训练结果。
- `src/utils/paths.py`：统一管理项目路径，例如 `DATA_DIR`、`RUNS_DIR`、`REPORTS_DIR`。
- `data/raw/`：原始数据，不建议提交到 Git。
- `data/processed/`：预处理后的 CSV，不建议提交到 Git。
- `runs/`：训练实验产物，每个实验一个目录，不建议提交到 Git。
- `reports/`：人工挑选或汇总后的图表、表格和报告。
- `comparison_fgm/`：BERT partial last 4/8 的有无 FGM 性能对比数据和图。
- `configs/`：后续可以放 YAML/JSON 配置文件。
- `docs/`：项目说明文档。

## 路径变化

旧目录和新目录的对应关系：

```text
models/<experiment>/      -> runs/<experiment>/
parameters/<experiment>/  -> runs/<experiment>/
outputs/comparison/       -> reports/model_comparison/
outputs/bert_freeze_*     -> runs/<experiment>/reports/
```

以后新训练产生的 checkpoint、tokenizer、config、metrics、history 会默认写入 `runs/<experiment_name>/`。

## FGM 对比

生成 partial last 4/8 的有无 FGM 对比：

```bash
python main.py --mode visualize --task fgm
```

输出文件：

- `comparison_fgm/bert_fgm_source_rows.csv`：从 `runs/` 读取到的 partial 4/8 原始实验行。
- `comparison_fgm/bert_fgm_comparison.csv`：有无 FGM 的宽表对比和差值。
- `comparison_fgm/bert_fgm_comparison.png`：测试集 Accuracy / Macro F1 对比图。
- `comparison_fgm/bert_fgm_gain.png`：FGM 相对 no FGM 的增益图。
- `comparison_fgm/bert_fgm_missing_runs.csv`：缺失的对照实验清单。
