# 操作记录

本文档记录本次项目整理、代码精简、评估补齐、图表刷新和文档更新的主要操作。为了以后复现方便，命令尽量按实际执行顺序写。

## 0. 操作流程和命令对照

这一节是给以后复现实验用的。左边是要做的事情，右边是应该运行的命令。

### 0.1 从零开始跑项目

| 步骤 | 要做什么 | 使用命令 | 说明 |
| --- | --- | --- | --- |
| 1 | 安装依赖 | `pip install -r requirements.txt` | 安装 jieba、torch、transformers、sklearn 等包。 |
| 2 | 检查主入口帮助 | `python main.py --help` | 查看项目支持哪些 `--mode`。 |
| 3 | 预处理原始数据 | `python main.py --mode data_processor` | 把 `data/raw/cnews.*.txt` 转成 `data/processed/*.csv`。 |
| 4 | 检查 processed 数据 | `python main.py --mode LR --check-data` | 检查 CSV 列名、标签是否合法，也会打印 train/val/test 数量。 |
| 5 | 训练 LR baseline | `python main.py --mode LR --small-grid` | 快速跑 TF-IDF + Logistic Regression，小网格比较快。 |
| 6 | 训练 LSTM | `python main.py --mode LSTM` | 训练 LSTM 模型，保存 history、metrics 和 checkpoint。 |
| 7 | 训练默认 BERT | `python main.py --mode BERT` | 默认是 full fine-tuning。CPU 会比较慢。 |
| 8 | 刷新总图表 | `python main.py --mode visualize --task both --bert-freeze-summary outputs\bert_freeze\_rebuild_from_configs.csv` | 生成 comparison、bert_freeze、fgm_comparison 下的图和 CSV。 |

### 0.2 `main.py --mode` 命令对应关系

| 命令 | 对应操作 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| `python main.py --mode data_processor` | 数据预处理 | `data/raw/cnews.train.txt`、`cnews.val.txt`、`cnews.test.txt` | `data/processed/train_data.csv`、`val_data.csv`、`test_data.csv` |
| `python main.py --mode LR --check-data` | 检查数据是否合法 | `data/processed/*.csv` | 终端打印检查结果 |
| `python main.py --mode LR --small-grid` | 快速训练 LR baseline | `data/processed/*.csv` | `configs/lr_tfidf/metrics.json`、`checkpoints/lr_tfidf/model.pkl` |
| `python main.py --mode LR` | 完整训练 LR baseline | `data/processed/*.csv` | `configs/lr_tfidf/`、`checkpoints/lr_tfidf/` |
| `python main.py --mode LSTM` | 训练 LSTM | `data/processed/*.csv` | `configs/lstm/`、`checkpoints/lstm/best_model.pth` |
| `python main.py --mode BERT` | 训练默认 BERT | `data/processed/*.csv` | `configs/bert_full_no_fgm/`、`checkpoints/bert_full_no_fgm/`、`outputs/bert_full_no_fgm/` |
| `python main.py --mode visualize --task models` | 生成 LR、LSTM、BERT 总体对比图 | `configs/lr_tfidf`、`configs/lstm`、`configs/bert_partial_last_8_no_fgm` | `outputs/comparison/` |
| `python main.py --mode visualize --task bert_freeze` | 生成 BERT 冻结策略图 | `outputs/bert_freeze/bert_freeze_summary.csv` 或 `configs/` | `outputs/bert_freeze/` |
| `python main.py --mode visualize --task fgm` | 生成 FGM 对比图 | `configs/bert_partial_last_4/8_*` | `outputs/fgm_comparison/` |
| `python main.py --mode visualize --task both` | 同时刷新 models 和 bert_freeze | `configs/` | `outputs/comparison/`、`outputs/bert_freeze/` |

### 0.3 BERT 常用命令

| 要做什么 | 使用命令 | 说明 |
| --- | --- | --- |
| 查看 BERT 参数 | `python main.py --mode BERT --help` | 会显示 BERT 子模块参数。 |
| full fine-tuning | `python main.py --mode BERT --finetune_strategy full` | 训练全部 BERT 参数。 |
| frozen BERT | `python main.py --mode BERT --finetune_strategy frozen` | 只训练分类头。 |
| partial last 4 | `python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4` | 只解冻最后 4 层。 |
| partial last 8 | `python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8` | 只解冻最后 8 层。 |
| partial last 4 + FGM | `python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4 --use-fgm` | 解冻最后 4 层和 embedding，并加入 FGM。 |
| partial last 8 + FGM | `python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8 --use-fgm` | 解冻最后 8 层和 embedding，并加入 FGM。 |
| 冻结策略批量实验 | `python main.py --mode BERT --freeze-sweep` | 批量跑 frozen、partial、full。 |
| 冻结策略 + FGM 批量实验 | `python main.py --mode BERT --freeze-sweep --use-fgm` | 会把 partial last 4/8 的 FGM 一起加入 sweep。 |
| 只评估已有 checkpoint | `python main.py --mode BERT --eval-only --experiment-name bert_partial_last_4_embedding_fgm` | 不训练，只读取已有 `best_model.pth` 生成评估结果。 |
| CPU 上加大评估 batch | `python main.py --mode BERT --eval-only --experiment-name bert_partial_last_4_embedding_fgm --eval-batch-size 32` | 只影响 eval-only 的 batch size。 |
| 评估并做温度校准 | `python main.py --mode BERT --eval-only --experiment-name bert_partial_last_4_embedding_fgm --eval-batch-size 32 --calibrate` | 同时生成 metrics、报告、错例、校准指标和可靠性图。 |

### 0.4 可视化命令

| 要生成什么 | 使用命令 | 输出目录 |
| --- | --- | --- |
| 三个模型总览图 | `python main.py --mode visualize --task models` | `outputs/comparison/` |
| BERT 冻结策略图 | `python main.py --mode visualize --task bert_freeze` | `outputs/bert_freeze/` |
| 从 configs 重建 BERT 冻结策略图 | `python main.py --mode visualize --task bert_freeze --bert-freeze-summary outputs\bert_freeze\_rebuild_from_configs.csv` | `outputs/bert_freeze/` |
| FGM 对比图 | `python main.py --mode visualize --task fgm` | `outputs/fgm_comparison/` |
| 一次性刷新全部常用图 | `python main.py --mode visualize --task both --bert-freeze-summary outputs\bert_freeze\_rebuild_from_configs.csv` | `outputs/comparison/`、`outputs/bert_freeze/`、`outputs/fgm_comparison/` |

### 0.5 本次补齐缺失结果的实际流程

| 顺序 | 操作 | 实际运行命令 | 结果 |
| --- | --- | --- | --- |
| 1 | 编译检查 | `python -m compileall -q main.py src models` | 检查 Python 语法是否正常。 |
| 2 | 检查数据 | `python main.py --mode LR --check-data` | 确认 train=50000、val=5000、test=10000。 |
| 3 | 补评估 partial last 4 FGM | `python main.py --mode BERT --eval-only --experiment-name bert_partial_last_4_embedding_fgm --eval-batch-size 32 --calibrate` | 生成 `metrics.json`、分类报告、混淆矩阵、预测、错例、校准图。 |
| 4 | 刷新所有图表 | `python main.py --mode visualize --task both --bert-freeze-summary outputs\bert_freeze\_rebuild_from_configs.csv` | 刷新 comparison、bert_freeze、fgm_comparison。 |
| 5 | 单独确认 FGM 对比 | `python main.py --mode visualize --task fgm` | 确认 partial last 4/8 的 with/without FGM 都齐了。 |
| 6 | 检查缺失文件是否还在 | `Test-Path outputs\fgm_comparison\bert_fgm_missing_runs.csv` | 返回 `False`，表示没有缺失项。 |

## 1. 初始检查

先查看项目结构和 Git 状态：

```bash
git status --short
Get-ChildItem -Force
Get-ChildItem -Recurse -File
```

## 2. 接入和精简 calibration.py

对 `src/calibration.py` 做了重写和精简，保留下面几类功能：

- 收集 BERT 输出 logits。
- 拟合 temperature scaling。
- 计算 ECE、NLL、accuracy。
- 保存校准指标 JSON。
- 保存校准分箱 CSV。
- 生成 reliability diagram。

相关输出位置：

```text
outputs/calibration/
|- <experiment>_calibration_metrics.json
|- <experiment>_calibration_bins.csv
`- <experiment>_reliability_diagram.png
```

## 3. 修改 BERT 训练和评估链路

修改了 `src/train_bert.py`：

- 新增 `--calibrate` 参数。
- 新增 `--calibration-bins` / `--calibration_bins` 参数。
- 在正常训练后的测试评估中接入 calibration。
- 在 `--eval-only` 评估中接入 calibration。
- 让 `predict_one_epoch()` 可以选择返回 logits，避免校准时重复跑完整 test set。
- 修复读取带 BOM 的 `config.json` 时报错的问题，将读取编码改为 `utf-8-sig`。

关键命令：

```bash
python main.py --mode BERT --eval-only --experiment-name bert_partial_last_4_embedding_fgm --eval-batch-size 32 --calibrate
```

这条命令只读取已有 checkpoint，不重新训练模型。

## 4. 补齐 partial last 4 FGM 评估

补齐的实验名：

```text
bert_partial_last_4_embedding_fgm
```

生成或更新的文件：

```text
configs/bert_partial_last_4_embedding_fgm/metrics.json

outputs/bert_partial_last_4_embedding_fgm/
|- bert_partial_last_4_embedding_fgm_classification_report.txt
|- bert_partial_last_4_embedding_fgm_confusion_matrix.png
|- bert_partial_last_4_embedding_fgm_predictions.csv
`- bert_partial_last_4_embedding_fgm_wrong_cases.csv

outputs/calibration/
|- bert_partial_last_4_embedding_fgm_calibration_metrics.json
|- bert_partial_last_4_embedding_fgm_calibration_bins.csv
`- bert_partial_last_4_embedding_fgm_reliability_diagram.png
```

评估结果：

```text
Test accuracy  = 0.9703
Test macro F1  = 0.970003
Test loss      = 0.101332
Wrong cases    = 297
```

温度校准结果：

```text
temperature = 1.164225
ECE before  = 0.012718
ECE after   = 0.008094
NLL before  = 0.101332
NLL after   = 0.095682
```

## 5. 刷新图表和汇总文件

运行：

```bash
python main.py --mode visualize --task both --bert-freeze-summary outputs\bert_freeze\_rebuild_from_configs.csv
python main.py --mode visualize --task fgm
```

刷新了下面这些目录：

```text
outputs/comparison/
outputs/bert_freeze/
outputs/fgm_comparison/
```

FGM 对比刷新后：

```text
partial last 4:
no FGM macro F1 = 0.966688
FGM macro F1    = 0.970003
delta           = +0.003315

partial last 8:
no FGM macro F1 = 0.969657
FGM macro F1    = 0.971554
delta           = +0.001897
```

确认缺失标记文件已经不存在：

```bash
Test-Path outputs\fgm_comparison\bert_fgm_missing_runs.csv
```

结果为：

```text
False
```

## 6. 精简 evaluate.py

修改 `src/evaluate.py`，删除旧的、当前主链路不用的函数：

- `calculate_classification_metrics`
- `save_metrics`
- `evaluate_classification_model`
- `merge_model_metrics`
- `plot_model_comparison`

保留当前仍然使用的函数：

- `save_classification_report`
- `plot_confusion_matrix`
- `save_predictions`
- `save_error_analysis`

精简后，`evaluate.py` 从约 390 行减少到约 165 行。

## 7. 合并 visualize.py 中重复逻辑

修改 `src/visualize.py`：

- 新增 `iter_bert_experiment_records()`，统一读取 `configs/` 下的 BERT 实验记录。
- `build_bert_freeze_summary_from_configs()` 和 `build_bert_fgm_summary_from_configs()` 共用这套读取逻辑。
- 新增 `first_not_none()`，避免原来用 `or` 回填指标时把合法的 `0` 当成空值。

这样可以减少 freeze summary 和 FGM summary 两条路径之间的重复代码。

## 8. 修复 main.py 的 help 转发

发现问题：

```bash
python main.py --mode BERT --help
```

原来只会显示 `main.py` 的总帮助，不会显示 BERT 子模块参数。

修复 `main.py` 后，现在可以正确显示子模块帮助：

```bash
python main.py --mode BERT --help
python main.py --mode visualize --help
```

## 9. README 重写

重写了根目录 `README.md`。

第一次重写偏正式项目文档；后来按要求改成更像大一学生写的实验说明，主要特点是：

- 语气更简单。
- 少用正式工程化表达。
- 增加“我对结果的简单理解”。
- 直接写当前实验结果。
- 保留必要运行命令。

## 10. STRUCTURE 文档更新

更新了：

```text
docs/PROJECT_STRUCTURE.md
```

新增生成：

```text
docs/PROJECT_STRUCTURE.docx
```

内容包括：

- 项目目录结构。
- 入口和模块职责。
- BERT、FGM、calibration 的产物流向。
- partial last 4 FGM 补齐后的结果。

DOCX 生成后做过结构检查：

```text
paragraphs = 61
tables     = 4
sections   = 1
headings   = 8
```

尝试用 LibreOffice 渲染 DOCX 为 PNG 时失败，原因是本机没有 `soffice`。之后尝试通过 `winget` 安装 LibreOffice，但安装器哈希不匹配，所以没有完成 PNG 视觉检查。

## 11. 已执行的验证命令

语法编译：

```bash
python -m compileall -q main.py src models
```

查看入口帮助：

```bash
python main.py --help
python main.py --mode BERT --help
python main.py --mode visualize --help
```

检查数据：

```bash
python main.py --mode LR --check-data
```

数据检查结果：

```text
Data check passed.
train=50000, val=5000, test=10000
```

刷新图表：

```bash
python main.py --mode visualize --task both --bert-freeze-summary outputs\bert_freeze\_rebuild_from_configs.csv
python main.py --mode visualize --task fgm
```

补齐 BERT partial last 4 FGM 评估：

```bash
python main.py --mode BERT --eval-only --experiment-name bert_partial_last_4_embedding_fgm --eval-batch-size 32 --calibrate
```

## 12. 当前仍然存在的说明

`bert_full_no_fgm` 当前有 `metrics.json`，但是没有完整的 `history.json`，所以在 BERT freeze summary 中 train 相关字段是空的。

这个缺口不能只靠 `eval-only` 补齐，因为 `history.json` 是训练过程中按 epoch 记录的。如果要补齐，需要重新训练 full BERT：

```bash
python main.py --mode BERT --finetune_strategy full
```

但是当前机器没有 CUDA，CPU 上重新训练 BERT 会很慢，所以这次没有重新训练 full BERT。

## 13. 当前主要结果文件

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

BERT 冻结策略：

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

校准结果：

```text
outputs/calibration/bert_partial_last_4_embedding_fgm_calibration_metrics.json
outputs/calibration/bert_partial_last_4_embedding_fgm_calibration_bins.csv
outputs/calibration/bert_partial_last_4_embedding_fgm_reliability_diagram.png
```
