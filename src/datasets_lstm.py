from torch.utils.data import Dataset,DataLoader
import torch
import jieba
from collections import Counter
try:
    from src.data_process import data_processor, build_id_map
except ModuleNotFoundError:
    from data_process import data_processor, build_id_map


def tokenize(text):
    return list(jieba.cut(text))


def build_vocab(text):
    counter = Counter()
    vocab = {
        "[PAD]":0,
        "[UNK]":1
    }
    for sentence in text:
        token = tokenize(sentence)
        counter.update(token)
    for tokens in counter:
        vocab[tokens] = len(vocab)

    return vocab

def vocab_transmission():
    train_data,_,_ = data_processor()
    vocab = build_vocab(train_data["text"])
    return vocab


def encode(vocab,text):
    tokens = tokenize(text)
    ids = [vocab.get(token,vocab["[UNK]"]) for token in tokens]
    return ids


def collate_fn(batch):

    text = [item[0] for item in batch]
    label = [item[1] for item in batch]

    max_len = max(len(sentence) for sentence in text)

    ids_list = []
    mask_list = []
    length_list = []

    for sentence in text:
        padding_len = max_len - len(sentence)
        padded_ids = sentence + [0] * padding_len

        attention_mask = [1] * len(sentence) + [0] * padding_len
        ids_list.append(padded_ids)
        mask_list.append(attention_mask)
        length_list.append(len(sentence))

    ids_tensor = torch.tensor(ids_list,dtype=torch.long)
    label_tensor = torch.tensor(label,dtype=torch.long)
    mask_tensor = torch.tensor(mask_list,dtype=torch.float)
    length_tensor = torch.tensor(length_list,dtype=torch.long)
    return ids_tensor,label_tensor,mask_tensor,length_tensor


class LSTMdataset(Dataset):
    def __init__(self,data,vocab):
        self.text = data["text"].tolist()
        self.label = build_id_map(data["label"])
        self.vocab = vocab

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        label = self.label[index]
        text_target = self.text[index]
        ids = encode(self.vocab,text_target)
        return ids, label


def process_loader(batch_size=16):
    train_data,val_data,test_data = data_processor()

    vocab = build_vocab(train_data["text"])
    
    train_dataset = LSTMdataset(train_data,vocab)
    test_dataset = LSTMdataset(test_data,vocab)
    val_dataset = LSTMdataset(val_data,vocab)

    train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,shuffle=False,batch_size=batch_size,collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,shuffle=False,batch_size=batch_size,collate_fn=collate_fn)

    return train_loader,val_loader,test_loader,vocab
