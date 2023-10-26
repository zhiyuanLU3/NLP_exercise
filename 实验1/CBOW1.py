import torch
import torch.nn as nn
import numpy as np
from data_set import MyDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import CBOW_model



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

text = MyDataset('output_strings_new.txt')
word = text.data
word = set(word)
word_size = len(word)

word_to_ix = {word: ix for ix, word in enumerate(word)}
ix_to_word = {ix: word for ix, word in enumerate(word)}

# 第二步：定义方法，自定义make_context_vector方法制作数据，自定义CBOW用于建立模型；

batch_size = 64  # 每个批次的大小
dataloader = DataLoader(text, batch_size=batch_size, shuffle=True)


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long, device=device)


def get_word_vector(word):
    word_idx = word_to_ix[word]
    word_tensor = torch.tensor([word_idx], dtype=torch.long, device=device)
    return model.embeddings(word_tensor).detach().numpy()


EMDEDDING_DIM = 16000  # 词向量维度

data = []
print(len(dataloader.dataset))
for i in range(2, len(dataloader) - 2):
    context = [dataloader.dataset[i - 2], dataloader.dataset[i - 1],
               dataloader.dataset[i + 1], dataloader.dataset[i + 2]]
    target = dataloader.dataset[i]
    data.append((context, target))

# 第三步：建立模型，开始训练；

model = CBOW_model.CBOW(word_size, EMDEDDING_DIM)
###
model.to(device=device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 第四步：开始训练
for epoch in range(10):
    total_loss = 0
    print(f'epoch: {epoch}\n')
    for text_ in data:
        context, target = text_
        context = [word_to_ix[i] for i in context]
        target = word_to_ix[target]
        context = torch.LongTensor(context).to(device)
        target = torch.LongTensor([target]).to(device)
        optimizer.zero_grad()
        out = model(context)

        loss = loss_function(out, target)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    print(f'loss value :{total_loss}')
torch.save(model.state_dict(), 'cbow_model.pth')

# print(f'文本数据: {" ".join(text)}\n')
