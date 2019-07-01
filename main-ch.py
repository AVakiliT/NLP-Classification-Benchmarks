import fairseq
import math
import os
import pickle
import random
from collections import defaultdict, Counter

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import gensim
import numpy as np
import torch
from scipy import sparse
from texttable import Texttable
from torch import nn
from torch.nn import functional as F, Parameter, init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Field, TabularDataset, BucketIterator, NestedField
from tqdm import tqdm

SEED = 1

torch.random.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
# %%
print('Reading Dataset...')

DATASET = 'agnews'
MAX_LEN = 60
N_EPOCHS = 12
NUM_CLASSES = 4

# DATASET = 'reuters50'
# MAX_LEN = 800
# N_EPOCHS = 25
# NUM_CLASSES = 50

# DATASET = 'yelp_full'
# MAX_LEN = 200
# N_EPOCHS = 4
# NUM_CLASSES = 5

# DATASET = 'ng20'
# MAX_LEN = 200
# N_EPOCHS = 18
# NUM_CLASSES = 20

MAX_WORD_LEN = 6
BATCH_SIZE = 32
LR = 1e-3
MIN_FREQ = 2
EMBEDDING_DIM = 50
EPSILON = 1e-13
INF = 1e13
HIDDEN_DIM = 100
PAD_FIRST = True
TRUNCATE_FIRST = False
SORT_BATCHES = False

print('Dataset ' + DATASET + ' loaded.')

# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CHAR = Field(sequential=True, use_vocab=True, tokenize=lambda x: list(x),
             batch_first=True, fix_length=MAX_WORD_LEN, pad_first=True)

WORD = NestedField(CHAR, tokenize=lambda x: x.split(), fix_length=MAX_LEN, pad_first=True, truncate_first=False)

LABEL = Field(sequential=False, use_vocab=False, batch_first=True)

columns = [('text', WORD),
           ('label', LABEL)]

train = TabularDataset(
    path=DATASET + '/train_clean.csv',
    format='csv',
    fields=columns,
    skip_header=True
)

test = TabularDataset(
    path=DATASET + '/test_clean.csv',
    format='csv',
    fields=columns,
    skip_header=True
)

WORD.build_vocab(train, min_freq=MIN_FREQ)

PAD = 1
# %%
train_iter = BucketIterator(
    train,
    BATCH_SIZE,
    device=device,
    repeat=False,
    shuffle=True,
    sort=SORT_BATCHES,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
)

test_iter = BucketIterator(
    test,
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


# %%
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        n_grams = [1,2]
        self.emb = nn.Embedding(len(CHAR.vocab), EMBEDDING_DIM, padding_idx=PAD)
        # self.char_cnns = nn.ModuleList([
        #     nn.Sequential(
        #         nn.ConstantPad1d((i // 2, (i - 1) // 2), 0),
        #         nn.Conv1d(EMBEDDING_DIM, EMBEDDING_DIM, i),
        #         nn.ReLU(),
        #         nn.MaxPool1d(MAX_WORD_LEN - i + 1)
        #     ) for i in n_grams
        # ])
        self.char_rnn = nn.GRU(EMBEDDING_DIM, EMBEDDING_DIM, batch_first=True)
        # word_dim = EMBEDDING_DIM * len(n_grams)
        word_dim = EMBEDDING_DIM
        self.rnn = nn.GRU(word_dim, word_dim, batch_first=True)
        self.final = nn.Sequential(
            nn.Linear(word_dim, word_dim // 2),
            nn.ReLU(),
            nn.Linear(word_dim // 2, NUM_CLASSES)
        )

    def forward(self, x):
        """

        :param xx: BSC
        :return:
        """

        xx = self.emb(x)

        # xx_new = []
        # for word in xx.permute(1, 0, 3, 2):
        #     a = torch.cat([n_gram(word).squeeze(-1) for n_gram in self.char_cnns], -1)
        #     xx_new.append(a)
        #
        # xx = torch.stack(xx_new, 1)


        xx = self.char_rnn(xx.view(-1, MAX_WORD_LEN, EMBEDDING_DIM))[0][:, -1, :].view(x.shape[0], MAX_LEN, -1)

        xx = self.rnn(xx)[0][:, -1, :]

        xx = self.final(xx)

        return xx




model = Model().to(device)


#%%
criterion = nn.CrossEntropyLoss(reduction='sum')

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

for i_epoch in range(1, N_EPOCHS + 1):
    model.train()
    loss_total = 0
    acc_total =0
    total = 0
    p_bar = tqdm(train_data_loader)
    for x, l in p_bar:
        optim.zero_grad()
        pred = model(x)
        loss = criterion(pred, l)
        loss.backward()
        optim.step()
        with torch.no_grad():
            acc = pred.argmax(-1).eq(l).sum().item()

        acc_total += acc
        total += l.shape[0]
        loss_total += loss.item()

        p_bar.set_description('TRAIN: {:.3f} {:.3f}'.format(acc_total / total, loss_total / total))

    with torch.no_grad():
        model.eval()
        loss_total = 0
        acc_total =0
        total = 0
        p_bar = tqdm(test_data_loader)
        for x, l in p_bar:
            pred = model(x)
            loss = criterion(pred, l)
            acc = pred.argmax(-1).eq(l).sum().item()

            acc_total += acc
            total += l.shape[0]
            loss_total += loss.item()

            p_bar.set_description('TEST: {:.3f} {:.3f}'.format(acc_total / total, loss_total / total))
