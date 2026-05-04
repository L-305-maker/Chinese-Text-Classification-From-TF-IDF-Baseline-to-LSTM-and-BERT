# 中文文本识别：从TF-IDF+LogisticRegressor到LSTM和BERT，比较不同模型的区别

本项目基于中文新闻文本分类数据集，比较传统机器学习模型与深度学习模型在十分类问题上的性能

## 项目目标
- 完成基础的文本处理流程，将纯文字样本处理为可供模型学习的形式
- 完成baseline：TF-IDF+LogisticRegressor模型的构建与参数调整
- 完成深度学习模型LSTM
- 完成深度学习模型BERT
- 比较不同模型的关键指标，例如f1_score，accuracy

## 数据集详细以及来源

### 数据集类别：采用中文新闻文本分类数据集，共有十个类别
- 体育
- 财经
- 游戏
- 家居
- 时政
- 房产
- 教育
- 时尚
- 科技
- 娱乐
### 数据集来源：[https://hyper.ai/cn/datasets/9277]
由于完整数据集过于庞大，本项目中只采用了部分数据来进行训练

## 项目结构
```text
Chinese Text Classification From TF-IDF Baseline to LSTM and BERT/
├─ data/
│  ├─ raw
│  ├─ processed
├─ models/
│  ├─ best_bert.pth
│  ├─ best_lstm.pth
│  ├─ log_tfidf_model.pkl
├─ notebooks
├─ outputs/
├─ src/
│  ├─ data_process.py
│  ├─ datasets_lstm.py
│  ├─ datasets_bert.py
│  ├─ evaluate.py
│  ├─ lr_train.py
│  ├─ train_bert.py
│  ├─ train_lstm.py
│  ├─ predict.py
├─ parameters/
│  ├─ best_config.json
│  ├─ best_lg_params.json
├─ requirements.txt
├─ README.md
├─ main.py
```