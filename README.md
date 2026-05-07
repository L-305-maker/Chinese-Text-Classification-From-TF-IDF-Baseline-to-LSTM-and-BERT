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
├─ data/  #由于文件过大，该文件夹并未上传
│  ├─ raw/
│  ├─ processed/
│  
├─ models/  #由于lstm_model.pth和bert_model.pth过大，也不做上传
│  ├─ bert/
│  ├─ lstm/
│  ├─ lr_tfidf/
│  
├─ notebooks
├─ outputs/
│  
├─ src/
│  ├─ data_process.py
│  ├─ datasets_lstm.py
│  ├─ datasets_bert.py
│  ├─ evaluate.py
│  ├─ lr_train.py
│  ├─ train_bert.py
│  ├─ train_lstm.py
│  ├─ predict.py
│  ├─ model_utils.py
│  
├─ parameters/
│  ├─ bert/
│  │  ├─ config.json
│  │  ├─ history.json
│  │  ├─ label_map.json
│  │  ├─ metrics.json
│  ├─ lr_tfidf/
│  │  ├─ best_params.json
│  │  ├─ config.json
│  │  ├─ label_map.json
│  │  ├─ metrics.json
│  ├─ lstm/
│  │  ├─ config.json
│  │  ├─ history.json
│  │  ├─ label_map.json
│  │  ├─ metrics.json
│  │  ├─ vocab.json
│  
├─ requirements.txt
├─ README.md
├─ main.py
```

## 项目详解
### LogisticRegressor+TF_IDF
- 模型使用pipeline将TF_IDF和LogisticREgressor进行流水线处理，然后用GridSearchCV得出最佳模型
### BERTClassifier
- 在这个环节，利用了bert-base-chinese模型，使用库自带的tokenizer模块处理文本，然后对文本进行一个处理与识别
### LSTMClassifier
- 此处在RNN的基础上加入了LSTM，加强了对文本的特征化处理，，即利用了LSTM的遗忘门与输入  
门等特性，增强了上下文关联对于文本识别的作用
- 此处需要自己完成collate_fn、tokenizer的编辑；且不同于单一的RNN，这里的batch不需要使用attention_mask，  
而是使用length，即每个文本的真实长度，因为我们利用pack_padded_sequence来进行文本的一个处理

## 项目亮点
- 引入argparse库，可以在命令行自主调参
- 将三个模型进行比对，比较不同模型在同一数据集下的表现
- 使用较为广泛的数据集(THUCnews)
- 对同一模型进行了多轮训练，并且设置了Early Stopping机制，节约训练时间
- 在main.py里设置了统一的接口，可以运行main.py，选择训练的模型
- 采用多个指标来多元化评估模型，例如accuracy和f1_score
