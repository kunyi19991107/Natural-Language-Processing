#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:08:50 2024

@author: yikun
"""

from tempfile import TemporaryDirectory
from typing   import Tuple
from torch    import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import math,os,torch
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#%%
class TransformerModel(nn.Module):
 
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        
        self.model_type  = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 定义编码器层
        encoder_layers           = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
 
        # 定义编码器，pytorch将Transformer编码器进行了打包，这里直接调用即可
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding           = nn.Embedding(ntoken, d_model)
        self.d_model             = d_model
        self.linear              = nn.Linear(d_model, ntoken)
 
        self.init_weights()
    
    # 初始化权重
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)
 
    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src     : Tensor, 形状为 [seq_len, batch_size]
            src_mask: Tensor, 形状为 [seq_len, seq_len]
        Returns:
            输出的 Tensor, 形状为 [seq_len, batch_size, ntoken]
        """
        src    = self.embedding(src) * math.sqrt(self.d_model)
        src    = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
#%%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        # 生成位置编码的位置张量
        position = torch.arange(max_len).unsqueeze(1)
        # 计算位置编码的除数项
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 创建位置编码张量
        pe = torch.zeros(max_len, 1, d_model)
        # 使用正弦函数计算位置编码中的奇数维度部分
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # 使用余弦函数计算位置编码中的偶数维度部分
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
 
    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, 形状为 [seq_len, batch_size, embedding_dim]
        """
        # 将位置编码添加到输入张量
        x = x + self.pe[:x.size(0)]
        # 应用 dropout
        return self.dropout(x)
#%%
from torchtext.datasets   import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab      import build_vocab_from_iterator
 
# 从torchtext库中导入WikiText2数据集
train_iter = WikiText2(split='train')
 
# 获取基本英语的分词器
tokenizer = get_tokenizer('basic_english')
 
# 通过迭代器构建词汇表
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
 
# 将默认索引设置为'<unk>'
vocab.set_default_index(vocab['<unk>'])
 
def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """将原始文本转换为扁平的张量"""
    data = [torch.tensor(vocab(tokenizer(item)), 
                         dtype=torch.long) for item in raw_text_iter]
    
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
 
# 由于构建词汇表时"train_iter"被使用了，所以需要重新创建
train_iter, val_iter, test_iter = WikiText2()
 
# 对训练、验证和测试数据进行处理
train_data = data_process(train_iter)
val_data   = data_process(val_iter)
test_data  = data_process(test_iter)
 
# 检查是否有可用的CUDA设备，将设备设置为GPU或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
def batchify(data: Tensor, bsz: int) -> Tensor:
    """将数据划分为 bsz 个单独的序列，去除不能完全容纳的额外元素。
    参数:
        data: Tensor, 形状为``[N]``
        bsz : int, 批大小
    返回:
        形状为 [N // bsz, bsz] 的张量
    """
    seq_len = data.size(0) // bsz
    data    = data[:seq_len * bsz]
    data    = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)
 
# 设置批大小和评估批大小
batch_size      = 20
eval_batch_size = 10
 
# 将训练、验证和测试数据进行批处理
train_data = batchify(train_data, batch_size)    # 形状为 [seq_len, batch_size]
val_data   = batchify(val_data, eval_batch_size)
test_data  = batchify(test_data, eval_batch_size)
#%%
bptt = 35
 
# 获取批次数据
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    参数:
        source: Tensor，形状为 ``[full_seq_len, batch_size]``
        i     : int，   当前批次索引
    返回:
        tuple (data, target)，
        - data形状为 [seq_len, batch_size]，
        - target形状为 [seq_len * batch_size]
    """
    # 计算当前批次的序列长度，最大为bptt，确保不超过source的长度
    seq_len = min(bptt, len(source) - 1 - i)
    
    # 获取data，从i开始，长度为seq_len
    data    = source[i:i+seq_len]
    
    # 获取target，从i+1开始，长度为seq_len，并将其形状转换为一维张量
    target  = source[i+1:i+1+seq_len].reshape(-1)
    
    return data, target
#%%
ntokens = len(vocab)  # 词汇表的大小
emsize  = 200         # 嵌入维度
d_hid   = 200         # nn.TransformerEncoder 中前馈网络模型的维度
nlayers = 2    #nn.TransformerEncoder中的nn.TransformerEncoderLayer层数
nhead   = 2           # nn.MultiheadAttention 中的头数
dropout = 0.2         # 丢弃概率
 
# 创建 Transformer 模型，并将其移动到设备上
model = TransformerModel(ntokens, 
                         emsize, 
                         nhead, 
                         d_hid, 
                         nlayers, 
                         dropout).to(device)
#%%
import time
 
criterion = nn.CrossEntropyLoss() # 定义交叉熵损失函数
lr        = 5.0  # 学习率
# 使用随机梯度下降（SGD）优化器，将模型参数传入优化器
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# 使用学习率调度器，每隔1个epoch，将学习率按0.95的比例进行衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
 
def train(model: nn.Module) -> None:
    model.train()  # 开启训练模式
    total_loss   = 0.
    log_interval = 200  # 每隔200个batch打印一次日志
    start_time   = time.time()
 
    num_batches = len(train_data) // bptt  # 计算总的batch数量
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
       
        data, targets = get_batch(train_data, i)  # 获取当前batch的数据和目标
        output        = model(data)               # 前向传播
        output_flat   = output.view(-1, ntokens)
        loss          = criterion(output_flat, targets)  # 计算损失
 
        optimizer.zero_grad()  # 梯度清零
        loss.backward()        # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 对梯度进行裁剪，防止梯度爆炸
        optimizer.step()       # 更新模型参数
 
        total_loss += loss.item()  # 累加损失值
        if batch % log_interval == 0 and batch > 0:
            
            lr = scheduler.get_last_lr()[0]  # 获取当前学习率
            
            # 计算每个batch的平均耗时
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval  
            cur_loss     = total_loss / log_interval  # 计算平均损失
            ppl          = math.exp(cur_loss)         # 计算困惑度
            
            # 打印日志信息
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            
            total_loss = 0            # 重置损失值
            start_time = time.time()  # 重置起始时间
 
def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # 开启评估模式
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)  # 获取当前batch的数据和目标
            seq_len       = data.size(0)             # 序列长度
            output        = model(data)              # 前向传播
            output_flat   = output.view(-1, ntokens)
            total_loss    += seq_len * criterion(output_flat, targets).item()  # 计算总损失
            
    return total_loss / (len(eval_data) - 1)  # 返回平均损失
#%%
best_val_loss = float('inf')  # 初始最佳验证损失为无穷大
epochs = 1  # 训练的总轮数
 
with TemporaryDirectory() as tempdir:  # 创建临时目录来保存最佳模型参数
    # 最佳模型参数的保存路径
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")  
 
    for epoch in range(1, epochs + 1):  # 遍历每个epoch
        epoch_start_time = time.time()  # 记录当前epoch开始的时间
        train(model)  # 进行模型训练
        val_loss = evaluate(model, val_data)  # 在验证集上评估模型性能，计算验证损失
        val_ppl = math.exp(val_loss)  # 计算困惑度
        elapsed = time.time() - epoch_start_time  # 计算当前epoch的耗时
        print('-' * 89)
        # 打印当前epoch的信息，包括耗时、验证损失和困惑度
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)
 
        if val_loss < best_val_loss:  # 如果当前验证损失比最佳验证损失更低
            best_val_loss = val_loss  # 更新最佳验证损失
            
            # 保存当前模型参数为最佳模型参数
            torch.save(model.state_dict(), best_model_params_path)  
 
        scheduler.step()  # 更新学习率
        
    # 加载最佳模型参数，即加载在验证集上性能最好的模型
    model.load_state_dict(torch.load(best_model_params_path))
#%%
test_loss = evaluate(model, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)