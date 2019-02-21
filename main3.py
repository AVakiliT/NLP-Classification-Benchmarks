import csv
from torch import nn
import nltk
import torch

from nltk.lm import MLE, Laplace, NgramCounter, KneserNeyInterpolated
from nltk.lm.api import LanguageModel
from nltk.lm.preprocessing import pad_both_ends, flatten
from nltk.lm.smoothing import _count_non_zero_vals
from nltk.util import bigrams, ngrams
import numpy as np
import math
import operator
import os
from collections import defaultdict, Counter
from functools import reduce, partial

import spacy
import nltk
import pandas as pd
from gensim.corpora import Dictionary
from sklearn.model_selection import train_test_split
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm
from nltk.collocations import BigramCollocationFinder

# nlp = spacy.load('en_core_web_sm')
from sklearn import preprocessing

from nltk import sent_tokenize, word_tokenize
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


#%%

# def f(root):
#     cs = []
#     ls = []
#     # root = 'C50/C50test'
#     for dire in os.listdir(root):
#         name = dire.split('-')[-1]
#         # topic = dire.split('-')[0]
#         course = os.path.join(root, dire)
#         for fname in os.listdir(course):
#             with open(os.path.join(course, fname)) as f:
#                 s = f.read()
#                 # s = [i.replace('\n', ' ').replace('"', '') for i in s if not str.startswith(i, '[')]
#                 # c.extend(s)
#                 s = s.replace('\n', ' ')
#                 cs.append(s)
#                 ls.append(name)
#                 # ts.extend([topic] * len(s))
#         # cs.extend(c)
#         # ls.extend(l)
#
#
#     df = pd.DataFrame(dict(text=cs, label=ls,
#                            # topic=ts
#                            ))
#     return df
#
#
# df = f('C50/C50train')
# df2 = f('C50/C50test')
# p = nltk.PorterStemmer()
# df.text = [' '.join([j for j in word_tokenize(i.lower().strip())]) for i in tqdm(df.text)]
# df2.text = [' '.join([j for j in word_tokenize(i.lower().strip())]) for i in tqdm(df2.text)]
# le = LabelEncoder()
# ls = le.fit_transform(df.label)
# df.label = ls
# ls2 = le.transform(df2.label)
# df2.label = ls2
#
# df.to_csv('train_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
# df2.to_csv('test_clean.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
#%%

BATCH_SIZE = 256
LR = 1e-3
N_EPOCHS = 100
MIN_FREQ = 8
EMB_DIM = 200
HIDDEN_DIM = 200
# folder = '/home/amir/IIS/Datasets/new_V2/Ubuntu_corpus_V2/'
# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


TEXT = Field(sequential=True, use_vocab=True,fix_length=200, tokenize=lambda x: x.split(), include_lengths=True,
             batch_first=True)

LABEL = Field(sequential=False, use_vocab=False, batch_first=True)

columns = [('text', TEXT),
           ('label', LABEL)]



train = TabularDataset(
    path='train_clean.csv',
    format='csv',
    fields=columns,
    skip_header=True
)


train = TabularDataset(
    path='test_clean.csv',
    format='csv',
    fields=columns,
    skip_header=True
)

TEXT.build_vocab(train, min_freq=MIN_FREQ)
#%%
train_iter = BucketIterator(
    train,
    BATCH_SIZE,
    device=device,
    repeat=False,
    shuffle=True,
    sort=True,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
)


test_iter = BucketIterator(
    train,
    BATCH_SIZE,
    device=device,
    repeat=False,
    shuffle=False,
    sort=True,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
)

class TrainIterWrap:

    def __init__(self, iterator) -> None:
        super().__init__()
        self.iterator = iterator

    def __iter__(self):
        for batch in self.iterator:
            yield batch.text, batch.label

    def __len__(self):
        return len(self.iterator)

train_data_loader = TrainIterWrap(train_iter)
test_data_loader = TrainIterWrap(test_iter)


#%%

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(TEXT.vocab), EMB_DIM, padding_idx=1)
        self.RNN = nn.GRU(EMB_DIM, HIDDEN_DIM, batch_first=True, bidirectional=False)
        self.final = nn.Linear(HIDDEN_DIM, 50, bias=False)

    def forward(self, x, x_len):
        x = self.emb(x)
        x = self.RNN(x)[0][:, -1, :]
        x = self.final(x).squeeze()
        return x


model = LSTMClassifier().to(device)

optimizer = torch.optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

progress_bar = tqdm(range(1, N_EPOCHS + 1))
for i_epoch in progress_bar:
    model.train()
    loss_total = 0
    accu_total = 0
    total = 0
    # progress_bar = tqdm(train_data_loader)
    for x, y in train_data_loader:
        optimizer.zero_grad()
        # mask = x[0] != PAD, x[1] != PAD
        prediction = model(x[0], x[1])
        loss = criterion(prediction, y.long())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (torch.argmax(prediction, 1).long() == y.long()).sum().item()

        batch_size = y.size(0)
        loss_total += loss.item()
        accu_total += acc
        total += batch_size

    model.eval()
    loss_total_test = 0
    accu_total_test = 0
    total_test = 0
    for x, y in test_data_loader:
        # mask = x[0] != PAD, x[1] != PAD
        prediction = model(x[0], x[1])
        loss = criterion(prediction, y.long())

        with torch.no_grad():
            acc = (torch.argmax(prediction, 1).long() == y.long()).sum().item()

        batch_size = y.size(0)
        loss_total_test += loss.item()
        accu_total_test += acc
        total_test += batch_size


    progress_bar.set_description(
        "[ LSS: {:.3f} ACC: {:.3f} ][ LSS: {:.3f} ACC: {:.3f} ]".format(
            loss_total / total,
            accu_total / total,
            loss_total_test / total_test,
            accu_total_test / total_test
        )
    )