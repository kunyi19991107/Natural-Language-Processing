#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 07:34:47 2024

@author: yikun
"""

# -*- coding: utf-8 -*-

#%%
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import os,PIL,pathlib,warnings
 
warnings.filterwarnings("ignore")             #忽略警告信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
print(device)
#%%
from torchtext.datasets import AG_NEWS
train_iter = AG_NEWS(split='train')      # 加载 AG News 数据集
#%%
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
 
tokenizer  = get_tokenizer('basic_english') # 返回分词器函数，训练营内“get_tokenizer函数详解”一文
 
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
 
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"]) # 设置默认索引，如果找不到单词，则会选择默认索引

#%%
text_pipeline  = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1
print(text_pipeline('here is the an example'))

#%%
from torch.utils.data import DataLoader
 
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    
    for (_label, _text) in batch:
        # 标签列表
        label_list.append(label_pipeline(_label))
        
        # 文本列表
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        
        # 偏移量，即语句的总词汇量
        offsets.append(processed_text.size(0))
        
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list  = torch.cat(text_list)
    offsets    = torch.tensor(offsets[:-1]).cumsum(dim=0) #返回维度dim中输入元素的累计和
    
    return label_list.to(device), text_list.to(device), offsets.to(device)
 
# 数据加载器
dataloader = DataLoader(train_iter,
                        batch_size=8,
                        shuffle   =False,
                        collate_fn=collate_batch)
#%%
from torch import nn
 
class TextClassificationModel(nn.Module):
 
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        
        self.embedding = nn.EmbeddingBag(vocab_size,   # 词典大小
                                         embed_dim,    # 嵌入的维度
                                         sparse=False) # 
        
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
 
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
 
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    
#%%
num_class  = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
em_size     = 64
model      = TextClassificationModel(vocab_size, em_size, num_class).to(device)
#%%
import time
 
def train(dataloader):
    model.train()  # 切换为训练模式
    total_acc, train_loss, total_count = 0, 0, 0
    log_interval = 500
    start_time   = time.time()
 
    for idx, (label, text, offsets) in enumerate(dataloader):
        
        predicted_label = model(text, offsets)
        
        optimizer.zero_grad()                    # grad属性归零
        loss = criterion(predicted_label, label) # 计算网络输出和真实值之间的差距，label为真实值
        loss.backward()                          # 反向传播
        optimizer.step()  # 每一步自动更新
        
        # 记录acc与loss
        total_acc   += (predicted_label.argmax(1) == label).sum().item()
        train_loss  += loss.item()
        total_count += label.size(0)
        
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:1d} | {:4d}/{:4d} batches '
                  '| train_acc {:4.3f} train_loss {:4.5f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count, train_loss/total_count))
            total_acc, train_loss, total_count = 0, 0, 0
            start_time = time.time()
 
def evaluate(dataloader):
    model.eval()  # 切换为测试模式
    total_acc, train_loss, total_count = 0, 0, 0
 
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            
            loss = criterion(predicted_label, label)  # 计算loss值
            # 记录测试数据
            total_acc   += (predicted_label.argmax(1) == label).sum().item()
            train_loss  += loss.item()
            total_count += label.size(0)
            
    return total_acc/total_count, train_loss/total_count
#%%
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
# 超参数
EPOCHS     = 10 # epoch
LR         = 5  # 学习率
BATCH_SIZE = 64 # batch size for training
 
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
 
train_iter, test_iter = AG_NEWS() # 加载数据
train_dataset = to_map_style_dataset(train_iter)
test_dataset  = to_map_style_dataset(test_iter)
num_train     = int(len(train_dataset) * 0.95)
 
split_train_, split_valid_ = random_split(train_dataset,
                                          [num_train, len(train_dataset)-num_train])
 
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
 
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    val_acc, val_loss = evaluate(valid_dataloader)
    
    if total_accu is not None and total_accu > val_acc:
        scheduler.step()
    else:
        total_accu = val_acc
    print('-' * 69)
    print('| epoch {:1d} | time: {:4.2f}s | '
          'valid_acc {:4.3f} valid_loss {:4.3f}'.format(epoch,
                                           time.time() - epoch_start_time,
                                           val_acc,val_loss))
 
    print('-' * 69)
#%%
print('Checking the results of test dataset.')
test_acc, test_loss = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(test_acc))
#%%

    