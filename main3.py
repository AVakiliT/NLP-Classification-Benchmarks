# import fairseq
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
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm


#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=3, type=int)
args = parser.parse_known_args()



#%%
SEED = 1

torch.random.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
# %%
print('Reading Dataset...')

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
#
DATASET = 'ng20'
MAX_LEN = 200
N_EPOCHS = 18
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
EMB_REQ_GRAD = True

print('Dataset ' + DATASET + ' loaded.')

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

TEXT.build_vocab(train, min_freq=MIN_FREQ)

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

def load_graph():
    df = pd.read_csv(DATASET + '/train_clean.csv')
    fname = 'stuff/' + DATASET + '_15_new.pmi.npy'
    if False:  # os.path.exists(fname):
        A = np.load(fname)
    else:
        def vectorize_string(s, stoi=TEXT.vocab.stoi):
            return [stoi[i] for i in s.split(' ')]

        train_vectors = df.text.apply(vectorize_string)

        n_docs = len(train_vectors)
        VOCAB_SIZE = len(TEXT.vocab.itos)

        word_to_class = np.zeros((VOCAB_SIZE, NUM_CLASSES))
        for s, l in (zip(train_vectors, df.label)):
            for w in set(s):
                word_to_class[w][l] += 1

        word_occur = word_to_class.sum(1)
        label_occur = Counter(df.label)

        def pmi(w, l):
            if word_to_class[w][l] == 0:
                return -1
            return (np.log2((word_to_class[w][l] * n_docs) / (word_occur[w] * label_occur[l]))) / np.log2(
                word_to_class[w][l] / n_docs)

        word_pmi = np.zeros((VOCAB_SIZE, NUM_CLASSES))
        for i in range(VOCAB_SIZE):
            for j in range(NUM_CLASSES):
                word_pmi[i][j] = pmi(i, j)

        wc = word_pmi[2:]
        wc[wc < 0] = 0
        # word_to_class = (word_to_class.T / word_to_class.sum(1)).T
        # word_to_class = ((word_to_class / word_to_class.sum(0)))

        A = np.zeros((VOCAB_SIZE - 2 + NUM_CLASSES, VOCAB_SIZE - 2 + NUM_CLASSES))
        # A[:VOCAB_SIZE - 2, :VOCAB_SIZE - 2] = pmi_m[2:, 2:]
        A[VOCAB_SIZE - 2:, :VOCAB_SIZE - 2] = wc.T
        A[:VOCAB_SIZE - 2, VOCAB_SIZE - 2:] = wc
        np.fill_diagonal(A, 1)
        np.save(fname, A)
    return A


# %%
def get_adj(A):
    D = A.sum(1) ** -0.5
    D = np.diag(D)

    AA = sparse.coo_matrix(A)
    DD_hi = sparse.coo_matrix(D)
    AADJ = DD_hi @ AA @ DD_hi
    return AADJ


def sparse2torch(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


# %%
print('Reading Embeddings...')
w2v = gensim.models.KeyedVectors.load_word2vec_format('/home/amir/IIS/Datasets/embeddings/glove.6B.100d.txt.w2vformat',
                                                      binary=True)

embedding_weights = torch.zeros(len(TEXT.vocab), EMBEDDING_DIM)
nn.init.normal_(embedding_weights)

unmatch = []
for i, word in enumerate(TEXT.vocab.itos):
    if word in w2v and i != PAD:
        embedding_weights[i] = torch.Tensor(w2v[word])
    else:
        unmatch.append(word)
        if i == PAD:
            embedding_weights[i] = torch.zeros(EMBEDDING_DIM)

print(len(unmatch) * 100 / len(TEXT.vocab.itos), '% of embeddings didn\'t match')

embedding_weights.to(device)


def get_emb():
    emb = nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=PAD, _weight=embedding_weights.clone())
    emb.weight.requires_grad = EMB_REQ_GRAD
    return emb


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


class TextCNN(nn.Module):

    def __init__(self):
        super().__init__()
        N_FILTERS = 50
        SIZES = [1, 3, 5]
        self.emb = get_emb()
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, N_FILTERS, (i, EMBEDDING_DIM)),
                nn.ReLU(),
                nn.MaxPool2d((MAX_LEN - i + 1, 1))
            )
            for i in SIZES
        ])
        self.final = nn.Linear(N_FILTERS * len(SIZES), NUM_CLASSES)

    def forward(self, x, _, __):
        x = self.emb(x)
        x = x.unsqueeze(1)
        xs = [l(x).squeeze() for l in self.cnn]
        x = torch.cat(xs, 1)
        return self.final(x).squeeze()


class TextCNN1d(nn.Module):

    def __init__(self):
        super().__init__()
        N_FILTERS = 50
        SIZES = [1,3,5]
        self.emb = get_emb()
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(EMBEDDING_DIM, N_FILTERS, i),
                nn.ReLU(),
                nn.MaxPool1d((MAX_LEN - i + 1))
            )
            for i in SIZES
        ])
        self.final = nn.Linear(N_FILTERS * len(SIZES), NUM_CLASSES)

    def forward(self, x, _, __):
        x = self.emb(x)
        x = x.transpose(1, 2)
        xs = [l(x).squeeze() for l in self.cnn]
        x = torch.cat(xs, 1)
        return self.final(x).squeeze()


class PytorchTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.transformer = nn.Transformer(EMBEDDING_DIM, EMBEDDING_DIM)
        self.final = nn.Linear(EMBEDDING_DIM, NUM_CLASSES)

    def forward(self, x, _, __):
        x = self.emb(x)
        x = self.transformer(x, x)
        xs = [l(x).squeeze() for l in self.cnn]
        x = torch.cat(xs, 1)
        return self.final(x).squeeze()

class TextCnnWithFusion(nn.Module):

    def __init__(self):
        super().__init__()
        N_FILTERS = 50
        SIZES = [1, 2, 3, 5]
        self.emb = get_emb()
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, N_FILTERS, (i, EMBEDDING_DIM)),
                nn.ReLU(),
                nn.MaxPool2d((MAX_LEN - i + 1, 1))
            )
            for i in SIZES
        ])
        self.final = nn.Linear(N_FILTERS * len(SIZES), NUM_CLASSES)

        self.cnn2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, N_FILTERS, (i, EMBEDDING_DIM)),
                nn.ReLU(),
                nn.MaxPool2d((MAX_LEN - i + 1, 1))
            )
            for i in SIZES
        ])
        self.final2 = nn.Linear(N_FILTERS * len(SIZES), NUM_CLASSES)

    def self_att(self, x, mask):
        s = (x @ x.transpose(1, 2))
        # zero att score for word on itself before softmax
        s = s * (1 - torch.eye(MAX_LEN, device=device))
        s = F.softmax(s, -1)
        # zero att score for word on itself again
        s = s * (1 - torch.eye(MAX_LEN, device=device))
        # mask zeros attention scores for pad tokens
        mask_2d = mask.unsqueeze(-1).float()
        mask_2d = (mask_2d @ mask_2d.transpose(1, 2))
        s = s * mask_2d
        # make sure each row sum is 1 and avoid divide by zero
        s = s / (s.sum(dim=-1, keepdim=True) + EPSILON)
        x_hat = s @ x
        return x_hat

    def forward(self, x, x_len, mask):
        x = self.emb(x)
        x_hat = self.self_att(x, mask)
        x_hat = x + x_hat
        x = x.unsqueeze(1)
        x_hat = x_hat.unsqueeze(1)
        xs = [l(x).squeeze() for l in self.cnn]
        x_hats = [l(x_hat).squeeze() for l in self.cnn2]
        x = torch.cat(xs, 1)
        x_hat = torch.cat(x_hats, 1)
        x = self.final(x).squeeze()
        x_hat = self.final2(x_hat).squeeze()
        return x + x_hat


def get_final(a, b, c=NUM_CLASSES):
    return nn.Sequential(
        nn.Linear(a, b),
        nn.ReLU(),
        nn.Linear(b, c),
    )


class TextCnnWithFusionAndContext(nn.Module):

    def __init__(self):
        super().__init__()
        N_FILTERS = 50
        SIZES = [1, 3, 5]
        HIDDEN_DIM = N_FILTERS * len(SIZES)
        self.emb = get_emb()
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, N_FILTERS, (i, EMBEDDING_DIM), padding=(i // 2, 0)),
                nn.ReLU(),
                # nn.MaxPool2d((MAX_LEN - i + 1, 1))
            )
            for i in SIZES
        ])

        self.rnn = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM // 2, bidirectional=True)

        self.finals = nn.ModuleList([
            get_final(EMBEDDING_DIM, 100, NUM_CLASSES),
            get_final(EMBEDDING_DIM, 100, NUM_CLASSES),
            get_final(HIDDEN_DIM, 100, NUM_CLASSES),
            get_final(HIDDEN_DIM, 100, NUM_CLASSES),
        ])

    def self_att(self, x, mask):
        s = (x @ x.transpose(1, 2))
        # zero att score for word on itself before softmax
        s = s * (1 - torch.eye(MAX_LEN, device=device))
        s = F.softmax(s, -1)
        # zero att score for word on itself again
        s = s * (1 - torch.eye(MAX_LEN, device=device))
        # mask zeros attention scores for pad tokens
        mask_2d = mask.unsqueeze(-1).float()
        mask_2d = (mask_2d @ mask_2d.transpose(1, 2))
        s = s * mask_2d
        # make sure each row sum is 1 and avoid divide by zero
        s = s / (s.sum(dim=-1, keepdim=True) + EPSILON)
        x_hat = s @ x
        return x_hat

    def forward(self, x, x_len, mask):
        x_emb = self.emb(x)
        x_hat = self.self_att(x_emb, mask)
        x_cnn = x_emb.unsqueeze(1)

        x_cnns = [l(x_cnn).squeeze() for l in self.cnn]

        x_cnn = torch.cat(x_cnns, -2).transpose(1, 2)

        x_rnn = self.rnn(x_emb)[0]

        representations = [x_emb, x_hat, x_cnn, x_rnn]

        representations = [i.max(-2)[0] for i in representations]

        representations = [l(i) for l, i in zip(self.finals, representations)]

        return sum(representations)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        FILTER_SIZE = 3
        POOLING_SIZE = 3
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=EMBEDDING_DIM, kernel_size=FILTER_SIZE),
                nn.ReLU(),
                nn.MaxPool1d(POOLING_SIZE)
            ) for _ in range(3)])
        self.final = nn.Linear(EMBEDDING_DIM, NUM_CLASSES)

    def forward(self, x, x_len, mask):
        x = self.emb(x)
        x = x.permute(0, 2, 1)
        for l in self.cnn:
            x = l(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze()
        x = self.final(x)
        return x


class SwemAvg(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.final = get_final(EMBEDDING_DIM, EMBEDDING_DIM // 2)

    def forward(self, x, x_len, mask):
        xx = self.emb(x)
        xx = xx.masked_fill(mask.unsqueeze(-1) ^ 1, 0)
        xx = xx.sum(-2)
        xx = xx / x_len.unsqueeze(1).float()
        xx = self.final(xx)
        return xx


class Swem_T2Sm(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.source = nn.Parameter(torch.ones(EMBEDDING_DIM) / math.sqrt(EMBEDDING_DIM))
        self.final = get_final(EMBEDDING_DIM, EMBEDDING_DIM // 2)

    def forward(self, x, x_len, mask):
        xx = self.emb(x)
        att_scores = torch.einsum('bse,e->bs', [xx, self.source])
        att_scores = att_scores.masked_fill(mask ^ 1, -INF)
        att_scores = F.softmax(att_scores, -1)
        xx = torch.einsum('bse,bs->be', [xx, att_scores])
        xx = xx / x_len.unsqueeze(1).float()
        xx = self.final(xx)
        return xx


class SwemMax(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.final = get_final(EMBEDDING_DIM, EMBEDDING_DIM // 2)

    def forward(self, x, x_len, mask):
        x = self.emb(x)
        x = x.masked_fill(mask.unsqueeze(-1) ^ 1, 0)
        x = x.max(-2)[0]
        x = self.final(x)
        return x


class SwemConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.final = get_final(EMBEDDING_DIM * 2, EMBEDDING_DIM // 2)

    def forward(self, x, x_len, mask):
        x = self.emb(x)
        x = x.masked_fill(mask.unsqueeze(-1) ^ 1, 0)
        x_avg = x.sum(-2)
        x_avg = x_avg / x_len.unsqueeze(1).float()
        x_max = x.max(-2)[0]
        x = torch.cat([x_avg, x_max], -1)
        x = self.final(x)
        return x


class SwemHier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        # self.emb.weight.requires_grad = False
        self.final = get_final(EMBEDDING_DIM, EMBEDDING_DIM // 2)

    def forward(self, x, x_len, mask):
        x = self.emb(x)
        x = F.adaptive_max_pool1d(F.avg_pool1d(x.transpose(1, 2), 5), 1).squeeze()
        x = self.final(x)
        return x


class RNN(nn.Module):
    def __init__(self, rnn=nn.LSTM, bidirectional=False, num_layers=1, bias=True):
        super().__init__()
        self.emb = get_emb()
        self.RNN = rnn(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=bidirectional,
                       num_layers=num_layers, bias=bias)
        self.final = get_final(EMBEDDING_DIM * (2 if bidirectional else 1), EMBEDDING_DIM // 2)

        self.name = ('Bi' if bidirectional else '') + type(self.RNN).__name__ + (
            'x' + str(num_layers) if num_layers > 1 else '')

    def forward(self, x, x_len, mask):
        x = self.emb(x)

        x = pack_padded_sequence(x, x_len, batch_first=True)
        x = self.RNN(x)[0]
        x = pad_packed_sequence(x, batch_first=True)[0]

        # x = x[:, -1, :]
        x = x[[torch.arange(0, x.shape[0]), x_len - 1]]

        x = self.final(x).squeeze()
        return x


class GhettoRNN(nn.Module):
    def __init__(self, rnn=nn.LSTM, bidirectional=False, num_layers=1):
        super().__init__()
        self.emb = get_emb()
        self.RNN = rnn(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True,
                       num_layers=num_layers, bias=True)
        if bidirectional:
            self.RNN_2 = rnn(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True,
                             num_layers=num_layers, bias=True)
        self.final = get_final(EMBEDDING_DIM * (2 if bidirectional else 1), EMBEDDING_DIM // 2)

        self.name = ('Bi' if bidirectional else '') + type(self.RNN).__name__ + (
            'x' + str(num_layers) if num_layers > 1 else '')

    def forward(self, x, x_len, mask):
        xx = self.emb(x)

        xx_1 = self.RNN(xx)[0]
        xx_1 = xx_1[[torch.arange(0, xx_1.shape[0]), x_len - 1]]

        xx_1 = self.final(xx_1).squeeze()
        return xx_1


class BiLSTMwithFusion(nn.Module):

    def __init__(self):
        super().__init__()
        HIDDEN_DIM = 200
        self.emb = get_emb()
        self.RNN = nn.GRU(2 * EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=1)
        self.RNN2 = nn.GRU(4 * HIDDEN_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=1)
        # self.final = nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM)
        self.final = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 200),
            nn.ReLU(),
            nn.Linear(200, NUM_CLASSES)
        )

    def self_attention(self, x, mask):
        a = torch.einsum('bne,bme->bnm', [x, x])
        a = a.masked_fill(torch.eye(MAX_LEN, device=device, dtype=torch.uint8), 0)
        a = F.softmax(a, -1)
        a = a.masked_fill(torch.eye(MAX_LEN, device=device, dtype=torch.uint8), 0)
        mask_2d = torch.einsum('bn,bm->bnm', [mask, mask])
        a = a.masked_fill(1 - mask_2d, 0)
        a = a / (a.sum(dim=-1, keepdim=True) + EPSILON)
        x_hat = torch.einsum('bne,bnm->bne', [x, a])
        return x_hat

    def forward(self, x, x_len, mask):
        x = self.emb(x)
        x_hat = self.self_attention(x, mask)
        x = torch.cat([x, x_hat], -1)
        x = self.RNN(x)[0]  # BNH

        x_hat2 = self.self_attention(x, mask)
        x = torch.cat([x, x_hat2], -1)
        x = self.RNN2(x)[0]
        # x = x[:, -1, :]
        x = x.max(-2)[0]
        x = self.final(x)
        return x


class AttentionClassifier(nn.Module):

    def self_att(self, x, mask):
        s = (x @ x.transpose(1, 2))
        # zero att score for word on itself before softmax
        s = s * (1 - torch.eye(MAX_LEN, device=device))
        s = F.softmax(s, -1)
        # zero att score for word on itself again
        s = s * (1 - torch.eye(MAX_LEN, device=device))
        # mask zeros attention scores for pad tokens
        mask_2d = mask.unsqueeze(-1).float()
        mask_2d = (mask_2d @ mask_2d.transpose(1, 2))
        s = s * mask_2d
        # make sure each row sum is 1 and avoid divide by zero
        s = s / (s.sum(dim=-1, keepdim=True) + EPSILON)
        x_hat = s @ x
        return x_hat

    def final_att(self, x, mask):
        # s = torch.tanh(self.W(x))
        # s = torch.tanh(x)
        s = torch.einsum('bwe,e->bw', [x, self.context_vector])
        s = F.softmax(s, -1)
        # mask zeros attention scores for pad tokens
        s = s * mask.float()
        # make sure each row sum is 1 and avoid divide by zero
        s = s / (s.sum(dim=-1, keepdim=True) + 1e-13)
        x_hat = torch.einsum('bwe,bw->be', [x, s])
        return x_hat

    def __init__(self, att, att2):
        super().__init__()
        self.att2 = att2
        self.att = att
        HIDDEN_DIM_2 = HIDDEN_DIM * (4 if att else 2)
        self.emb = get_emb()
        self.RNN = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.RNN2 = nn.GRU(HIDDEN_DIM_2, HIDDEN_DIM_2, batch_first=True, bidirectional=True)
        self.W = nn.Linear(HIDDEN_DIM_2 * 2, HIDDEN_DIM_2 * 2)
        self.context_vector = Parameter(torch.randn(HIDDEN_DIM_2 * 2), requires_grad=True)
        self.final = nn.Linear(HIDDEN_DIM_2 * 2, NUM_CLASSES)

    def forward(self, x, x_len, mask):
        x = self.emb(x)
        x = self.RNN(x)[0]

        if self.att:
            x_hat = self.self_att(x, mask)
            # concat attention vector for each word to original vector
            x = torch.cat([x, x_hat], -1)

        x = self.RNN2(x)[0]

        if self.att2:
            x = self.final_att(x, mask)
        else:
            # x = x[:, -1, :]
            x = x.mean(-2)
        x = self.final(x).squeeze()
        return x


class leam(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        # self.label_vectors = nn.Parameter(torch.randn(NUM_CLASSES, EMBEDDING_DIM))
        self.label_vectors = nn.Parameter(
            (torch.Tensor(np.tile(np.eye(NUM_CLASSES) / math.sqrt(EMBEDDING_DIM), EMBEDDING_DIM // NUM_CLASSES))))
        # nn.init.uniform_(self.label_vectors, -1/math.sqrt(EMBEDDING_DIM), 1/math.sqrt(EMBEDDING_DIM))
        self.conv = nn.Conv1d(NUM_CLASSES, NUM_CLASSES, kernel_size=51, stride=1, padding=25)
        self.final = get_final(EMBEDDING_DIM, EMBEDDING_DIM // 2)

    def forward(self, x, x_len, mask):
        xx = self.emb(x)
        g = torch.einsum('ce,bse->bcs', [self.label_vectors, xx])
        # g_hat = self.label_vectors.norm(dim=-1, keepdim=True) @ x.norm(dim=-1, keepdim=True).transpose(-2, -1)
        g_hat = torch.einsum('c,bs->bcs', [self.label_vectors.norm(dim=-1), xx.norm(dim=-1)])
        g_hat[g_hat == 0] = EPSILON
        g = g / g_hat
        u = F.relu(self.conv(g))  # BCS
        # u =g
        m = u.max(1)[0]  # BS
        b = m.masked_fill(mask ^ 1, -INF)
        b = F.softmax(b, -1)
        z = torch.einsum('bse,bs->be', [xx, b])

        z = self.final(z)

        # z = (torch.einsum('be,ce->bc', [z, self.label_vectors]) / torch.einsum('b,c->bc', [z.norm(dim=-1), self.label_vectors.norm(dim=-1)]))

        return z


class LeamWithGraphEmbed(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        # self.label_vectors = nn.Parameter(torch.randn(NUM_CLASSES, EMBEDDING_DIM))
        self.label_vectors = nn.Parameter(
            (torch.Tensor(np.tile(np.eye(NUM_CLASSES) / math.sqrt(EMBEDDING_DIM), EMBEDDING_DIM // NUM_CLASSES))),
            requires_grad=False)
        # # nn.init.uniform_(self.label_vectors, -1/math.sqrt(EMBEDDING_DIM), 1/math.sqrt(EMBEDDING_DIM))
        self.conv = nn.Conv1d(NUM_CLASSES, NUM_CLASSES, kernel_size=51, stride=1, padding=25)
        self.final = get_final(EMBEDDING_DIM, EMBEDDING_DIM // 2)
        self.gcn = GCN(EMBEDDING_DIM, EMBEDDING_DIM, EMBEDDING_DIM, 0)

    def forward(self, x, x_len, mask):
        x = self.emb(x)

        ##
        inp = torch.cat([self.emb.weight[2:], self.label_vectors])
        out = self.gcn(inp, adj)  # (V-2+C)E
        new_lv = out[-NUM_CLASSES:]
        # new_lv = F.relu(self.lv2lv(new_lv))
        # new_lv = self.label_vectors
        ##

        g = torch.einsum('ce,bse->bcs', [new_lv, x])
        # g_hat = self.label_vectors.norm(dim=-1, keepdim=True) @ x.norm(dim=-1, keepdim=True).transpose(-2, -1)
        g_hat = torch.einsum('c,bs->bcs', [new_lv.norm(dim=-1),
                                           x.norm(dim=-1)])
        g_hat[g_hat == 0] = 1
        g = g / g_hat
        u = F.relu(self.conv(g))  # BCS
        # u =g
        m = F.max_pool1d(u.transpose(-2, -1), NUM_CLASSES).squeeze(-1)  # BS
        b = m.masked_fill(mask ^ 1, -INF)
        b = F.softmax(b, -1)
        z = torch.einsum('bse,bs->be', [x, b])

        z = self.final(z)

        return z


class RnnWithAdditiveSourceAttention(nn.Module):
    class SourceAdditiveAttention(nn.Module):

        def __init__(self, dim):
            super().__init__()
            self.W1 = nn.Linear(dim, dim, bias=False)
            # self.W2 = nn.Parameter(nn.zeros(dim, dim))
            # init.uniform_(self.W2, -1 / math.sqrt(dim), 1 / math.sqrt(dim))
            self.W = nn.Linear(dim, 1, bias=False)
            # self.Wfs = nn.Linear(dim, dim)
            # self.Wfh = nn.Linear(dim, dim, bias=False)

        def forward(self, keys, values, mask):
            keys = torch.tanh(self.W1(keys))  # BLE
            scores = self.W(keys).squeeze(-1)  # BL
            scores = scores.masked_fill(mask, -INF)
            scores = F.softmax(scores, -1)
            s = torch.einsum('bl,ble->be', [scores, values])
            # f = torch.sigmoid(self.Wfs(s) + self.Wfh(values)) ##
            return s

    class SourceMultiplicativeAttention(nn.Module):

        def __init__(self, dim):
            super().__init__()
            self.source = nn.Parameter(torch.zeros(dim))
            nn.init.uniform_(self.source, -1 / math.sqrt(dim), 1 / math.sqrt(dim))
            self.W = nn.Parameter(torch.zeros(dim, dim))
            bound = 1 / math.sqrt(dim)
            init.uniform_(self.W, -bound, bound)

        def forward(self, keys, values, mask):
            scores = torch.einsum('ble,ee,e->bl', [keys, self.W, self.source])  # BL
            scores = scores.masked_fill(mask, -INF)
            scores = F.softmax(scores, -1)
            s = torch.einsum('bl,ble->be', [scores, values])
            return s

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.RNN = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.att = self.SourceAdditiveAttention(EMBEDDING_DIM * 3)
        self.final = get_final(HIDDEN_DIM * 2, HIDDEN_DIM // 2)
        self.name = 'BiGRU_T2S+'

    def forward(self, x, x_len, mask):
        x = self.emb(x)

        xh = self.RNN(x)[0]
        # xx = xh.mean(-2)
        xx = torch.cat([x, xh], -1)
        xx = self.att(xx, xh, mask ^ 1)
        xx = self.final(xx)
        return xx


class DiSAN(nn.Module):

    def __init__(self, hidden_dim=EMBEDDING_DIM):
        super().__init__()
        self.emb = get_emb()
        self.Wh = nn.Linear(hidden_dim, hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.uniform_(self.b, -1 / math.sqrt(hidden_dim), 1 / math.sqrt(hidden_dim))
        self.c = nn.Parameter(torch.Tensor([5.0]), requires_grad=False)
        self.fw_mask = torch.ByteTensor(np.tri(MAX_LEN, MAX_LEN, dtype='uint8')). \
            unsqueeze(-1).expand(MAX_LEN, MAX_LEN, hidden_dim).to(device)
        self.bw_mask = torch.ByteTensor(np.tri(MAX_LEN, MAX_LEN, dtype='uint8')).t(). \
            unsqueeze(-1).expand(MAX_LEN, MAX_LEN, hidden_dim).to(device)
        self.Wf1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wf2 = nn.Linear(hidden_dim, hidden_dim)
        self.Ws1 = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.Ws = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.final = get_final(hidden_dim * 2, hidden_dim)

    def multi_dim_masked_attention(self, h, att, m):
        att = att.masked_fill(m, -INF)
        att = F.softmax(att, -2)  # BLLE
        s = torch.einsum('bme,blme->ble', [h, att])  # BLE
        f = torch.sigmoid(self.Wf1(s) + self.Wf2(h))  # BLE
        u = f * h + (1 - f) * s
        return u

    def forward(self, x, _, mask):
        x = self.emb(x)  # BLE
        h = F.elu(self.Wh(x))
        h1 = self.W1(h)
        h2 = self.W2(h)
        att = self.c * torch.tanh((h1.unsqueeze(2) + h2.unsqueeze(1) + self.b) / self.c)  # BLLE
        mask_2d = ((mask.unsqueeze(1) ^ 1).__or__(mask.unsqueeze(2)) ^ 1).unsqueeze(-1)  # LL1
        att = att.masked_fill(mask_2d, -INF)  # BLLE
        u_fw = self.multi_dim_masked_attention(h, att, self.fw_mask)  # BLE
        u_bw = self.multi_dim_masked_attention(h, att, self.bw_mask)  # BLE
        u = torch.cat([u_fw, u_bw], -1)  # BL(2E)

        att_s = self.Ws(F.elu(self.Ws1(u)))  # BL(2E)
        s_s = (u * att_s).sum(-2)  # B(2E)
        return self.final(s_s)  # BC


class DeepConv(nn.Module):
    def __init__(self, filter_size, filter_shape, latent_size):
        super(DeepConv, self).__init__()
        self.embed = get_emb()
        self.convs1 = nn.Conv1d(EMBEDDING_DIM, filter_size, filter_shape, stride=2)
        self.bn1 = nn.BatchNorm1d(filter_size)
        self.convs2 = nn.Conv1d(filter_size, filter_size * 2, filter_shape, stride=2)
        self.bn2 = nn.BatchNorm1d(filter_size * 2)

        self.len_1 = (MAX_LEN - (filter_shape - 1)) // 2  # 28
        self.len_2 = (self.len_1 - (filter_shape - 1)) // 2

        self.convs3 = nn.Conv1d(filter_size * 2, latent_size, self.len_2, stride=1)

        self.final = get_final(latent_size, latent_size // 2)

        # weight initialize for conv layer
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x, _, __):
        x = self.embed(x)

        # reshape for convolution layer
        x = x.transpose(1, 2)

        h1 = F.relu(self.bn1(self.convs1(x)))
        h2 = F.relu(self.bn2(self.convs2(h1)))
        h3 = F.relu(self.convs3(h2))

        o = self.final(h3.squeeze(-1))
        return o


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class Transformer(nn.Module):
    class MultiHeadAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.l_q = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
            self.l_k = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
            self.l_v = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=False)
            self.norm = nn.LayerNorm((MAX_LEN, EMBEDDING_DIM))
            self.ff = nn.Sequential(
                nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
                nn.ReLU(),
                nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
            )

        def attention(self, q, k, v, mask2d):
            q, k, v = self.l_q(q), self.l_k(k), self.l_v(v),
            sims = ((q @ k.transpose(1, 2)) / math.sqrt(EMBEDDING_DIM))  # bmn
            sims = sims.masked_fill(mask2d, -INF)
            sims = sims.softmax(-1)
            o = torch.einsum('bmn,bne->bme', [sims, v])
            return o

        def forward(self, xx, mask2d):
            xx = self.norm(xx + self.attention(xx, xx, xx, mask2d))
            # xx = self.norm(xx + self.att(xx, xx, xx)[0])
            xx = self.norm(xx + self.ff(xx))
            return xx

    def __init__(self):
        super().__init__()
        self.emb = get_emb()
        self.pe = PositionalEncoder(EMBEDDING_DIM, MAX_LEN)
        self.att = Transformer.MultiHeadAttention()
        # self.att = nn.MultiheadAttention(EMBEDDING_DIM, 1)

        self.final = get_final(EMBEDDING_DIM, EMBEDDING_DIM // 2)

    def forward(self, x, x_len, __):
        mask = x == PAD  # bm
        mask2d = mask.unsqueeze(1) | mask.unsqueeze(2)
        xx = self.emb(x)
        xx = self.pe(xx)
        xx = self.att(xx, mask2d)
        xx = xx.sum(1)
        xx = xx / x_len.unsqueeze(1).float()
        xx = self.final(xx)
        return xx


# class DynaConv(nn.Module):
#     class Block(nn.Module):
#
#         def __init__(self, k):
#             super().__init__()
#             self.dc1 = fairseq.modules.DynamicConv1dTBC(EMBEDDING_DIM, kernel_size=k, padding_l=k // 2, num_heads=10,
#                                                         weight_softmax=True)
#             self.l1 = nn.Linear(EMBEDDING_DIM, 2 * EMBEDDING_DIM)
#             self.l2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
#             self.ff = nn.Sequential(
#                 nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM),
#                 nn.ReLU(),
#                 nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
#             )
#             self.norm = nn.LayerNorm((EMBEDDING_DIM))
#
#         def forward(self, x):
#             x = self.norm(x + self.dc1(F.glu(self.l1(x))))
#             x = self.norm(x + self.ff(x))
#             return x
#
#     def __init__(self):
#         super().__init__()
#         self.emb = get_emb()
#         self.b1 = self.Block(30)
#         self.final = get_final(EMBEDDING_DIM, EMBEDDING_DIM // 2)
#
#     def forward(self, x, _, __):
#         xx = self.emb(x)
#         xx = xx.transpose(0, 1).contiguous()
#         xx = self.b1(xx)
#         xx = xx.sum(0)
#         xx = xx / x_len.unsqueeze(1).float()
#         xx = self.final(xx)
#         return xx


# %%
# print('building graph')
# A = load_graph()
# print('building adj')
# ADJ = get_adj(A)
# ADJ = sparse.coo_matrix(ADJ)
# adj = sparse2torch(ADJ).to(device)
# INP = sparse2torch(sparse.coo_matrix(np.eye(*adj.shape))).to(device)
# %%

# model = AttentionClassifier(False, False)
# model = AttentionClassifier(False, True)
# model = AttentionClassifier(True, False)
# model = AttentionClassifier(True, True)
# model = CNN()
# model = TextCnnWithFusionAndContext()
# model = leam2()
# model = leam()
# model = DiSAN()
# model = SwemAvg(); N_EPOCHS = N_EPOCHS * 2
# model = SwemMax(); N_EPOCHS = N_EPOCHS * 2
# model = SwemConcat();
# N_EPOCHS = N_EPOCHS * 2
# model = SwemHier()
# model = TextCNN()
model = TextCNN1d()
# model = TextCnnWithFusion()
# model = RNN(nn.GRU, bidirectional=True, num_layers=1, bias=True); N_EPOCHS = N_EPOCHS * 2
# model = GhettoRNN(nn.LSTM, bidirectional=False, num_layers=1); N_EPOCHS = N_EPOCHS * 2
# model = RNN(nn.GRU, bidirectional=False, num_layers=1)
# model = RNN(nn.GRU, bidirectional=True, num_layers=1)
# model = BiLSTMwithFusion()
# model = RnnWithAdditiveSourceAttention()
# model = LeamWithGraphEmbed()
# model = Swem_T2Sm(); N_EPOCHS *= 2
# model = DeepConv(EMBEDDING_DIM, 5, EMBEDDING_DIM * 2)
# model = DynaConv()
# model = Transformer()
# model = PytorchTransformer()

# N_EPOCHS = 25

model = model.to(device)

criterion = nn.CrossEntropyLoss(reduction='sum')

metrics_history_all = []

optimizer = torch.optim.Adam(model.parameters())
metrics_history = []
progress_bar = tqdm(range(1, N_EPOCHS + 1))
for i_epoch in progress_bar:
    model.train()
    loss_total = 0
    accu_total = 0
    total = 0
    # progress_bar = tqdm(train_data_loader)
    for (x, x_len), y in train_data_loader:
        optimizer.zero_grad()
        mask = x != PAD
        prediction = model(x, x_len, mask)
        loss = criterion(prediction, y.long())

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (torch.argmax(prediction, 1).long() == y.long()).sum().item()

        batch_size = y.size(0)
        loss_total += loss.item()
        accu_total += acc
        total += batch_size

    with torch.no_grad():
        model.eval()
        loss_total_test = 0
        accu_total_test = 0
        total_test = 0
        for (x, x_len), y in test_data_loader:
            mask = x != PAD
            prediction = model(x, x_len, mask)
            loss = criterion(prediction, y.long())

            with torch.no_grad():
                acc = (torch.argmax(prediction, 1).long() == y.long()).sum().item()

            batch_size = y.size(0)
            loss_total_test += loss.item()
            accu_total_test += acc
            total_test += batch_size

    metrics = (
        loss_total / total,
        accu_total / total,
        loss_total_test / total_test,
        accu_total_test / total_test
    )
    progress_bar.set_description(
        "[ TRAIN LSS: {:.3f} ACC: {:.3f} ][ TEST LSS: {:.3f} ACC: {:.3f} ]".format(*metrics)
    )
    metrics_history.append(metrics)

if os.path.exists('histories/' + DATASET):
    with open('histories/' + DATASET, 'rb') as f:
        baselines = pickle.load(f)
else:
    baselines = dict()


def model_name(model):
    try:
        a = model.name
    except:
        a = type(model).__name__
    return a


baselines[model_name(model)] = metrics_history

# metrics_history_all_t = np.array(list(baselines.values())).transpose((0, 2, 1))
# metrics_history_all_t = np.nan_to_num(metrics_history_all_t, 0)
metrics_history_all_t = [np.array(i).T for i in baselines.values()]
model_names = list(baselines.keys())

rows = [(n, np.round(v[1].max(), 3), np.round(v[3].max(), 3)) for n, v in zip(model_names, metrics_history_all_t)]
rows = sorted(rows, key=lambda x: x[2])
t = Texttable()
t.add_rows(rows, header=False)
t.header(('Model', 'Acc Train', 'Acc Test'))
print(t.draw())
with open('metrics/' + DATASET + '.txt', 'w') as f:
    print(t.draw(), file=f)
# %%
sns.set()
plt.clf()
ax = plt.gca()
for history in metrics_history_all_t:
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(history[1], color=color)
    plt.plot(history[3], '--', color=color)

# plt.yticks([i / 10 for i in range(11)])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

legends = []
for i in model_names:
    for j in ['train', 'test']:
        legends.append(' '.join((i, j)))

plt.legend(legends, loc='upper left',
           bbox_to_anchor=(0, -0.2),
           fancybox=True, shadow=True, ncol=2)
plt.title("Comparison of ANN Models for " + DATASET)
# plt.show()
plt.savefig('plots/' + DATASET + '.png', dpi=300, bbox_inches='tight')
# %%
with open('histories/' + DATASET, 'wb') as f:
    pickle.dump(baselines, f)
