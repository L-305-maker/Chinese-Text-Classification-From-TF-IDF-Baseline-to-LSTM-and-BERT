# Chinese Text Classification from Baseline TF-IDF to LSTM and BERT

> 项目结构已整理，源码位于src/，模型定义位于src/models/

本项目基于中文新闻文本分类数据集，完成了从传统机器学习模型到深度学习模型、与训练语言模型微调的历程；当前代码支持：

- 在`data_process.py`中实现数据预处理，将原始数据统一转化为.csv文件
- 传统机器学习模型：`TF-IDF + LogisticRegressor`
- 深度学习模型：`LSTM`文本分类器
- 预训练模型`bert-base-chinese`文本分类器
- BERT冻结策略实现：`frozen/partial/full`
- BERT可训练参数统计：`trainable ratio/trainable parameters/ total parameters`
- 训练曲线、混淆矩阵、模型对比、classification report、错例输出等

## 1.项目任务：

这是一个中文文本多分类项目，输入是一段中文文本，输出是新闻所属分类，标签映射存储在`src/model_utils.py`

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
- `main.py`：程序的统一入口
- `src/data_process.py`：负责读取原始的文本数据，并输出为.csv文件，存储在data/processed文件夹下
- `src/lr_train.py`：训练TF-IDF + LogisticRegressor
- `src/train_lstm.py`：训练LSTM
- `src/train_bert.py`：训练BERT，同时支持冻结策略，参数统计和FGM
- `src/adversarial.py`：FGM对抗生成训练的实现
- `src/evaluate.py`：classification report、confusion martix、预测结果、错例保存
- `src/visualize.py`：指标汇总与结果可视化
- `parameters/`：保存config、metrics、history、label map等
- `models`：保存模型的权重、tokenizer等等
- `outputs/`：保存汇总.csv文件、图像报告、预测与错例分析等

## 3、环境安装
建议使用Python 3.9+
```bash
conda create -n textcls python=3.10
conda activate textcls
pip install requirements.txt
```

如果使用GPU加速训练，请根据本机CUDA版本安装适配的Pytorch版本
首次运行 BERT 时，`transformers` 可能需要下载 `bert-base-chinese`，如果本地没有缓存，需要保证网络可访问 Hugging Face。

## 4、统一入口

项目通过`main.py`选择不同模式
```bash
python main.py --mode data_processor/Log_TF_IDF/LSTM/BERT/visualize
```

## 5、训练TF-IDF + LogisticRegressor

快速检查数据
```bash
python main.py --mode Log_TF_IDF --check-data
```
使用较小参数网络进行训练
```bash
python main.py --mode Log_TF_IDF --small-grid
```
使用部分训练样本进行调试
```bash
python main.py --mode Log_TF_IDF --sample-size 5000 --small-grid
```
输出结果会专门存储到`parameters/和outputs/`下的专门文件夹下

实现细节：
- 使用`jieba`分词
- 使用`TfidfVectorizer`提取文本特征
- 使用`LogisticRegressor(solver="saga")`训练分类器
- 使用`GridSearchCV`搜索最佳参数
---

## 6、训练LSTM

常用参数
```bash
python main.py --mode LSTM --batch_size 16 --embed_dim 64 --hidden_size 512
```
可配置参数：
| 参数 | 默认值 |
|:---|---|
| `--batch_size` | 16 |
| `--embed_dim` | 64 |
| `--num_layers` | 1 |
| `--dropout` | 0.3 |
| `--hidden_size` | 512 |
| `--learning_rate` | 0.001 |
| `--epochs` | 5 |

输出结果会专门存储到`parameters/和outputs/`下的专门文件夹下

---

## 7、训练BERT
分类器定义位于`models/bert/bert_ubfreeze_last_n_layers.py`，训练逻辑位于`src/train_bert.py`

### 7.1 各微调策略示例
```bash
python main.py --mode BERT --finetune_stragety full/partial/frozen …………
```
### 7.2 可配置参数：
| 参数 | 默认值 |
|:---|:---|
| `--dropout` | 0.3 |
| `--max_len` | 256 |
| `--batch_size` | 16 |
| `--epochs` | 5 |
| `--finetune_stragety` | full/partial/frozen |
| `--unfreeze_last_n_layers` | None |
| `--bert_lr` | 2e-5 |
| `--classifier_lr` | 1e-4 |
| `--weight_decay` | 0.01 |
### 7.3 一次性运行多种冻结策略
```bash
python main.py --mode BERT --freeze-sweep
```
sweep会汇总保存`outputs/bert_freeze_summary.csv`，里面包括个实验模型的具体情况

### 7.4 partial-4/8可以使用FGM
FGM是现在`adversarial.py`，当前设计并不只是对frozen embedding进行简单扰动，而是单独定义了两类实验：
```text
partial-4 + Embedding-unfrozen + FGM
partial-8 + Embedding-unfrozen + FGM 
```
开启FGM-sweep的指令如下：
```bash
python main.py --mode BERT --freeze-sweep --use-fgm …………
```
FGM只允许在以下条件中使用
```text
finetune_strategy = partial
unfreeze_last_n_layers in {4，8}
```
使用FGM与否的对比结果会储存在`comparison_fgm/`

## 8、各可视化处理
默认读取`parameters/`下各模型的`metrics.json`和`history.josn`
```bash
python main.py --mode visualize
```
指定模型进行可视化操作：
```bash
python main.py --mode visualize --models lr_tfidf lstm bert_full_no_fgm
```
指定可视化模型对比和BERT freeze sweep
```bash
python main.py --mode visualize --task both --bert-freeze-summary outputs/bert_freeze_summary.csv
```

---

## 9、错误分析评估
`src/evaluate.py` 提供以下能力：

- accuracy / precision / recall / F1 计算。
- `classification_report` 保存为 txt。
- `confusion_matrix` 保存为 png。
- 预测结果保存为 CSV。
- 错误样本保存为 CSV。

其中wrong cases适合继续做error analysis，比较具有哪些特征的文本更容易混淆

---

## 10、实验流程推荐

```bash
python main.py --mode data_processor
python main.py --mode Log_TF_IDF --small-grid
python main.py --mode LSTM --epochs 5 --batch_size 16
python main.py --mode BERT --finetune_strategy full --epochs 3 --batch_size 8 --max_len 128
python main.py --mode BERT --freeze-sweep --use-fgm --fgm-epsilon 1.0 --epochs 3 --batch_size 8 --max_len 128
python main.py --mode visualize --task both --bert-freeze-summary outputs/bert_freeze_summary.csv
```

## 11、Git 与大文件建议

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

## 12、项目特点

- 从传统机器学习、RNN 到 BERT，覆盖不同复杂度的中文文本分类方案。
- 所有模型共用同一份 processed CSV 和统一 label map，便于公平比较。
- BERT 支持 full、frozen、partial 三类微调策略。
- freeze sweep 自动记录 trainable ratio 和 trainable parameters，便于分析性能与训练成本的关系。
- Partial-4/8 + Embedding Unfrozen + FGM 提供成对对比实验，能直接观察对抗训练对测试集表现的影响。
- 结果文件按 `models/`、`parameters/`、`outputs/` 分层保存，便于复现和分析。