# 项目运行操作指南

本文档说明当前代码在 `online_shopping_10_cats` 十分类数据集上的完整运行方式。所有命令建议在项目根目录执行：

```powershell
cd "D:\python\Chinese Text Classification From TF-IDF Baseline to LSTM and BERT"
```

## 1. 入口

统一入口是：

```powershell
python main.py --mode <任务名> [任务参数]
```

| `--mode` | 对应代码 | 作用 |
| --- | --- | --- |
| `data_processor` | `src.data_processor.process_raw_to_csv()` | 从 `online_shopping_10_cats.csv` 生成 processed CSV |
| `LR` | `src.train_lr.main()` | 训练或检查 `TF-IDF + Logistic Regression` |
| `Log_TF_IDF` | `src.train_lr.main()` | `LR` 的别名 |
| `LSTM` | `src.train_lstm.main()` | 训练 LSTM 文本分类模型 |
| `BERT` | `src.train_bert.main()` | 训练、评估 BERT，支持 freeze sweep、embedding-only 对照、FGM、校准 |
| `visualize` | `src.visualize.main()` | 根据已有实验产物生成对比图表 |

查看帮助：

```powershell
python main.py --help
python main.py --mode LR --help
python main.py --mode BERT --help
```

## 2. 环境

```powershell
pip install -r requirements.txt
```

主要依赖：

| 依赖 | 用途 |
| --- | --- |
| `pandas` | 读取和保存 CSV |
| `scikit-learn` | TF-IDF、Logistic Regression、数据划分和指标 |
| `jieba` | LR 和 LSTM 的中文分词 |
| `torch` | LSTM/BERT 训练与评估 |
| `transformers` | `bert-base-chinese` tokenizer 和 BERT 模型 |
| `matplotlib` | 图表输出 |

BERT 首次运行需要本地已有 `bert-base-chinese` 缓存，或允许 `transformers` 联网下载。

## 3. 数据

当前数据集是 `online_shopping_10_cats`，原始文件位置：

```text
data/raw/online_shopping_10_cats/online_shopping_10_cats.csv
```

原始列含义：

| 列 | 当前项目用途 |
| --- | --- |
| `cat` | 十分类标签 |
| `review` | 输入文本 |
| `label` | 原始情感标签，本项目不使用 |

当前类别：

```text
书籍、平板、手机、水果、洗发水、热水器、蒙牛、衣服、计算机、酒店
```

生成 processed 数据：

```powershell
python main.py --mode data_processor
```

输出文件：

| 输出 | 内容 |
| --- | --- |
| `data/processed/train_data.csv` | 训练集，约 70% |
| `data/processed/val_data.csv` | 验证集，约 15% |
| `data/processed/test_data.csv` | 测试集，约 15% |

处理规则：

- 使用 `cat -> label`，`review -> text`
- 丢弃空文本或空标签
- 校验所有标签都存在于 `src/model_utils.py` 的 `LABEL2ID`
- 使用分层划分，保证每个 split 尽量保留类别分布
- 输出 CSV 编码为 `utf-8-sig`

检查 processed 数据：

```powershell
python main.py --mode LR --check-data
```

该命令只检查数据，不训练模型。

## 4. 推荐流程

从零开始：

```powershell
pip install -r requirements.txt
python main.py --mode data_processor
python main.py --mode LR --check-data
python main.py --mode LR --small-grid
python main.py --mode LSTM --epochs 1
```

确认数据和轻量模型链路正常后，再运行 BERT：

```powershell
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8
```

## 5. LR / TF-IDF

```powershell
python main.py --mode LR [参数]
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--check-data` | 只检查 processed 数据 |
| `--sample-size 1000` | 从训练集分层抽样，适合快速调试 |
| `--small-grid` | 只跑一组较小参数 |

常用命令：

```powershell
python main.py --mode LR --check-data
python main.py --mode LR --sample-size 1000 --small-grid
python main.py --mode LR --small-grid
```

输出：

```text
checkpoints/lr_tfidf/model.pkl
configs/lr_tfidf/config.json
configs/lr_tfidf/best_params.json
configs/lr_tfidf/validation_search_results.json
configs/lr_tfidf/metrics.json
configs/lr_tfidf/label_map.json
```

## 6. LSTM

```powershell
python main.py --mode LSTM [参数]
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--epochs` | 训练轮数 |
| `--batch-size` | batch size |
| `--tokenizer word|char` | 分词粒度 |
| `--sample-size` | 分层抽样训练 |
| `--max-len` | 最大序列长度 |
| `--max-vocab-size` | 词表上限 |
| `--bidirectional` / `--no-bidirectional` | 是否使用双向 LSTM，默认开启 |
| `--class-weight-power` | 类别权重平滑指数，默认 `0.5` |
| `--no-class-weights` | 关闭类别权重 |

调试命令：

```powershell
python main.py --mode LSTM --epochs 1 --sample-size 1000
```

当前 LSTM 默认参数偏向性能优先：`embed_dim=256`、`hidden_size=256`、`num_layers=2`、双向 LSTM、`dropout=0.35`、`epochs=20`、`max_len=256`、`max_vocab_size=80000`，并默认启用平滑类别权重以缓解类别不均衡。

输出：

```text
checkpoints/lstm/best_model.pth
configs/lstm/config.json
configs/lstm/history.json
configs/lstm/metrics.json
configs/lstm/vocab.json
configs/lstm/label_map.json
```

## 7. BERT

```powershell
python main.py --mode BERT [参数]
```

常用训练方式：

```powershell
python main.py --mode BERT
python main.py --mode BERT --finetune_strategy full
python main.py --mode BERT --finetune_strategy frozen
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4 --no-fgm --unfreeze-embeddings
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8 --no-fgm --unfreeze-embeddings
```

`--unfreeze-embeddings` 用于在 partial last 4/8 的基础上额外解冻 BERT embedding 层。若要做“只额外解冻 embedding、不启用 FGM”的对照实验，需要同时加上 `--no-fgm`；对应实验名会保存为 `bert_partial_last_4_embedding_no_fgm` 或 `bert_partial_last_8_embedding_no_fgm`。

当前 BERT 默认参数偏向性能优先：`finetune_strategy=partial`、`unfreeze_last_n_layers=8`、`dropout=0.2`、`epochs=6`、`bert_lr=2e-5`、`classifier_lr=2e-4`、`fgm_epsilon=0.8`、`label_smoothing=0.02`。当策略为 partial last 4/8 时，默认自动启用 FGM；可用 `--no-fgm` 关闭。

FGM 只支持 partial last 4/8：

```powershell
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 4 --use-fgm
python main.py --mode BERT --finetune_strategy partial --unfreeze_last_n_layers 8 --use-fgm
```

BERT 也默认启用平滑类别权重，可通过 `--no-class-weights` 关闭，通过 `--class-weight-power` 调整权重强度。

批量 freeze sweep：

```powershell
python main.py --mode BERT --freeze-sweep
python main.py --mode BERT --freeze-sweep --no-fgm
```

默认 sweep 会包含 partial last 4/8 的 embedding-only 对照和 FGM 版本；使用 `--no-fgm` 时会跳过 FGM，但仍保留 partial last 4/8 的 embedding-only 无 FGM 对照。

仅评估已有 checkpoint：

```powershell
python main.py --mode BERT --eval-only --experiment-name bert_partial_last_8_no_fgm
```

切换数据集后，不建议用旧 checkpoint 做新数据集评估；旧 checkpoint 的类别语义来自旧训练数据。

## 8. 可视化

```powershell
python main.py --mode visualize --task models
python main.py --mode visualize --task bert_freeze
python main.py --mode visualize --task fgm
python main.py --mode visualize --task both
```

注意：`visualize` 读取 `configs/` 和 `outputs/` 中已有实验结果。数据集切换后，必须先重新训练并生成新指标，再使用这些图表做结论。

## 9. 数据集切换后的注意事项

- 上一版新闻分类原始数据文件已删除。
- `data_processor` 只面向 `online_shopping_10_cats`。
- 当前代码的 `num_classes` 仍为 10，无需修改模型输出维度。
- 旧 `checkpoints/`、`configs/`、`outputs/` 中的指标不代表当前数据集表现。
- 正式对比 LR/LSTM/BERT 前，应依次重跑数据处理、数据检查和各模型训练。
