import math
import os
import pickle
import random
from collections import defaultdict, Counter
from itertools import combinations

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import gensim
import numpy as np
import torch
from scipy import sparse
from scipy.linalg import fractional_matrix_power
from texttable import Texttable
from torch import nn
from torch.nn import functional as F, Parameter, init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm

from new_utils import sliding_window

SEED = 2
torch.random.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
# %%
# DATASET = 'agnews'
# MAX_LEN = 60
# N_EPOCHS = 12
# NUM_CLASSES = 4

# DATASET = 'reuters50'
# MAX_LEN = 800
# N_EPOCHS = 25
# NUM_CLASSES = 50

# DATASET = 'yelp_full'
# MAX_LEN = 200
# N_EPOCHS = 4
# NUM_CLASSES = 5

DATASET = 'ng20'
MAX_LEN = 200
N_EPOCHS = 25
NUM_CLASSES = 20

BATCH_SIZE = 32
LR = 1e-3
MIN_FREQ = 8
EMBEDDING_DIM = 100
EPSILON = 1e-13
INF = 1e13
HIDDEN_DIM = 100
PAD_FIRST = True
TRUNCATE_FIRST = False
SORT_BATCHES = False

# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TEXT = Field(sequential=True, use_vocab=True, fix_length=MAX_LEN, tokenize=lambda x: x.split(),
             include_lengths=True,
             batch_first=True, pad_first=PAD_FIRST, truncate_first=TRUNCATE_FIRST)

LABEL = Field(sequential=False, use_vocab=False, batch_first=True)

columns = [('text', TEXT),
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

TEXT.build_vocab(train, test, min_freq=MIN_FREQ)

PAD = 1


# %%

def load_graph_and_labels():
    df = pd.read_csv(DATASET + '/train_clean.csv')
    df_test = pd.read_csv(DATASET + '/test_clean.csv')
    labels = df.label
    labels_test = df_test.label
    fname = 'stuff/' + DATASET + '_fixed.txtgcn.npy'
    if False: #os.path.exists(fname):
        m_all = np.load(fname)
    else:
        def vectorize_string(s, stoi=TEXT.vocab.stoi):
            return [stoi[i] for i in s.split(' ')]

        train_vectors = df.text.apply(vectorize_string).to_list()
        test_vectors = df_test.text.apply(vectorize_string).to_list()

        n_docs = len(train_vectors) + len(test_vectors)
        VOCAB_SIZE = len(TEXT.vocab.itos)

        # co_occur = np.full((VOCAB_SIZE, VOCAB_SIZE), 0)
        # occur = np.zeros(VOCAB_SIZE)
        # n_windows = 0
        # for v in tqdm(train_vectors + test_vectors):
        #     for window in sliding_window(v, 15):
        #         n_windows += 1
        #         window = set(window)
        #         for i in window:
        #             occur[i] += 1
        #         for i, j in combinations(window, 2):
        #             co_occur[i, j] += 1
        #             co_occur[j, i] += 1
        #
        #
        # def pmi(i, j):
        #     if co_occur[i, j] == 0:
        #         return -INF
        #     pmi = np.log2((co_occur[i][j] * n_docs) / (occur[i] * occur[j]))
        #     # npmi
        #     return pmi
        #
        # pmi_m = np.zeros((VOCAB_SIZE, VOCAB_SIZE))
        # for i in tqdm(range(2, len(TEXT.vocab.itos))):
        #     for j in range(2, len(TEXT.vocab.itos)):
        #         pmi_m[i, j] = pmi(i, j)
        #
        # pmi_c = pmi_m[2:, 2:]
        # # np.fill_diagonal(pmi_c, 1)
        #
        # pmi_c[pmi_c < 0] = 0

        inv = defaultdict(Counter)
        for d, v in enumerate(tqdm(train_vectors + test_vectors)):
            for w in v:
                inv[w][d] += 1

        tfidf_m = np.zeros((VOCAB_SIZE, n_docs))
        for w, dc in inv.items():
            for d, c in dc.items():
                tfidf_m[w][d] = c / len(dc)

        m_all = np.zeros((VOCAB_SIZE - 2 + n_docs, VOCAB_SIZE - 2 + n_docs))
        # m_all[:(VOCAB_SIZE - 2), :(VOCAB_SIZE - 2)] = pmi_c
        m_all[(VOCAB_SIZE - 2):, :(VOCAB_SIZE - 2)] = tfidf_m[2:].T
        m_all[:(VOCAB_SIZE - 2), (VOCAB_SIZE - 2):] = tfidf_m[2:]
        np.fill_diagonal(m_all, 1)

        np.save(fname, m_all)
    return m_all, labels, labels_test


A, train_labels, test_labels = load_graph_and_labels()
train_labels = torch.LongTensor(train_labels).to(device)
test_labels = torch.LongTensor(test_labels).to(device)


# %%

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = input @ self.weight
        output = adj @ support
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


model = GCN(A.shape[0], 200, NUM_CLASSES, 0.5).to(device)


# %%

def get_adj(A):
    D = A.sum(1) ** -0.5
    D = np.diag(D)

    AA = sparse.coo_matrix(A)
    DD_hi = sparse.coo_matrix(D)
    AADJ = DD_hi @ AA @ DD_hi
    return AADJ


ADJ = get_adj(A)
ADJ = sparse.coo_matrix(ADJ)


# %%

def sparse2torch(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


optim = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

x = sparse2torch(sparse.coo_matrix(np.eye(*A.shape))).to(device)
adj = sparse2torch(ADJ).to(device)

for i in range(N_EPOCHS):
    model.train()
    optim.zero_grad()

    out = model(x, adj)
    prediction = out[(len(TEXT.vocab.itos) - 2):(len(TEXT.vocab.itos) - 2 + len(train_labels))]
    loss = criterion(prediction, train_labels)
    loss.backward()
    optim.step()
    acc = (prediction.argmax(1) == train_labels).float().mean().item()
    print(i, 'trn', np.round(loss.item(), 3), np.round(acc, 3))
    model.eval()
    prediction_test = out[(len(TEXT.vocab.itos) - 2 + len(train_labels)):]
    loss = criterion(prediction_test, test_labels)

    acc = (prediction_test.argmax(1) == test_labels).float().mean().item()
    print(i, 'tst', np.round(loss.item(), 3), np.round(acc, 3))
