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
from sklearn.cluster import KMeans
from spacy.lang import punctuation
from spacy.lang.en import stop_words
from texttable import Texttable
from torch import nn
from torch.nn import functional as F, Parameter, init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm

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

DATASET = 'ng20'
MAX_LEN = 200
N_EPOCHS = 10
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

print('Dataset ' + DATASET + ' loaded.')

# rem = stop_words.STOP_WORDS.union({'.', ',', '"', ':', ';', '-'})
rem = {}

# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TEXT = Field(sequential=True, use_vocab=True, fix_length=MAX_LEN, tokenize=lambda x: [i for i in x.split() if i not in rem],
             # include_lengths=True,
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
    return nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=PAD, _weight=embedding_weights.clone())


# %%
train_list = list(train_data_loader)
all_train_text = torch.cat([x[0] for x in train_list], 0)
del train_list


def random_train_sample(n):
    return all_train_text[np.random.randint(0, len(all_train_text), n)]


# %%

def sentence_emb_avg(xx, mask):
    return xx.max(1)[0]
    len_s = (mask ^ 1).long().sum(1, keepdim=True).float()
    y_s = xx.sum(1) / len_s
    return y_s


class Aspect(nn.Module):
    def __init__(self, emb_dim, asp_dim, inf):
        super().__init__()
        self.emb = get_emb()
        self.asp_dim = asp_dim
        self.emb_dim = emb_dim
        self.inf = inf
        # self.M = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.S = nn.Parameter(torch.randn(emb_dim))
        self.W = nn.Linear(emb_dim, asp_dim)
        self.T = nn.Parameter(torch.randn(asp_dim, emb_dim))  #
        self.conv = nn.Conv1d(asp_dim, asp_dim, kernel_size=51, stride=1, padding=25)
        self.reset_params()

    def reset_params(self):
        bound = 1 / math.sqrt(self.emb_dim)
        # init.uniform_(self.M, -bound, bound)
        init.uniform_(self.S, -bound, bound)

        X = self.emb.weight.detach().cpu().numpy()
        kc = KMeans(n_jobs=8, n_clusters=self.asp_dim)
        kc_fit = kc.fit(X)
        centroids = kc_fit.cluster_centers_
        self.T = nn.Parameter(torch.FloatTensor(centroids))

    def encode(self, xx, mask):
        """

        :param xx: BSE
        :param mask: BS
        :return: BT
        """

        y_s = sentence_emb_avg(xx, mask)

        z_s, a_i = self.weighted_avg(xx, mask, y_s)

        p_t_ = self.W(z_s)  # BA
        p_t = p_t_.softmax(-1)

        return p_t, z_s, a_i, p_t_

    def weighted_avg(self, xx, mask, y_s=None):
        # d_i = torch.einsum('bse,be->bs', [xx, y_s])
        # d_i.masked_fill_(mask, -INF)
        # a_i = F.softmax(d_i, -1)
        # z_s = torch.einsum('bse,bs->be', [xx, a_i])
        # return z_s, a_i
        # len_s = (mask ^ 1).long().sum(1, keepdim=True).float()
        # y_s = xx.sum(1) / len_s
        # return y_s
        g = torch.einsum('ae,bse->bas', [self.T, xx])
        g_hat = torch.einsum('a,bs->bas', [self.T.norm(dim=-1), xx.norm(dim=-1)])
        g_hat[g_hat == 0] = EPSILON
        g = g / g_hat
        u = F.relu(self.conv(g))
        m = u.max(1)[0]  # BS
        a_i = m.masked_fill(mask, -INF)
        a_i = F.softmax(a_i, -1)
        z_s = torch.einsum('bse,bs->be', [xx, a_i])
        return z_s, a_i

    def decode(self, p_t):
        r_s = p_t @ self.T
        return r_s

    def forward(self, x):
        mask = x == PAD
        x = self.emb(x)
        p_t, z_s, a_i, p_t_ = self.encode(x, mask)
        r_s = self.decode(p_t)
        return p_t, z_s, r_s, a_i, p_t_


class Classifier(nn.Module):

    def __init__(self, aspect_dim):
        super().__init__()
        self.aspect_dim = aspect_dim
        self.final = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM // 2),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM // 2, NUM_CLASSES)
        )

    def forward(self, p_t):
        return self.final(p_t)


class AspectExtractor(nn.Module):

    def __init__(self, aspects):
        super().__init__()
        self.aspect = Aspect(EMBEDDING_DIM, aspects, INF)
        self.classifier = Classifier(aspects)

    def forward(self, x):
        p_t, z_s, r_s, a_i, p_t_ = self.aspect(x)
        out = self.classifier(z_s)
        return r_s, z_s, a_i, p_t, out


def margin_loss(reconstruction, representation, negatives=None):
    """
    :param reconstruction: BE
    :param representation: BE
    :param negatives: BME
    :return: j: B
    """

    if negatives is not None:
        m = negatives.shape[1]
        # r_sim = torch.einsum('be,be->b', [reconstruction, representation]).repeat(m, 1)
        # n_sim = torch.einsum('be,bme->mb', [reconstruction, negatives])

        r_sim_ = F.cosine_similarity(reconstruction, representation, dim=-1).repeat(m, 1)
        n_sim_ = F.cosine_similarity(reconstruction.unsqueeze(1).repeat(1, m, 1), negatives, dim=-1).t()

        # r_sim_ = (reconstruction.norm(dim=-1) * representation).repeat(m, 1)
        # n_sim_ = torch.einsum('b,bm->mb', [reconstruction.nomr(dim=-1), negatives.norm(dim=-1)])
    else:
        r_sim = torch.einsum('be,be->b', [reconstruction, representation])
        n_sim = 0

    # j = F.relu(1 - r_sim + n_sim).sum()
    j = F.relu(1 - r_sim_ + n_sim_).sum()

    if negatives is not None:
        j = j / m

    return j


def reg_loss(t):
    """

    :param t: ET
    :return: 1
    """
    t_dot_tt = (t @ t.t()) / (t.norm(dim=-1, keepdim=True) @ t.norm(dim=-1, keepdim=True).t())
    t_dot_tt = t_dot_tt - torch.eye(t_dot_tt.shape[0]).to(device)
    u = t_dot_tt.norm()
    return u


# %%

model = AspectExtractor(20)

model = model.to(device)

c_loss = nn.CrossEntropyLoss(reduction='sum')

metrics_history_all = []

optimizer = torch.optim.Adam(model.parameters())
metrics_history = []
progress_bar = tqdm(range(1, N_EPOCHS + 1))
for i_epoch in progress_bar:
    model.train()
    loss_1_total = 0
    loss_2_total = 0
    loss_3_total = 0
    accu_total = 0
    total = 0
    # progress_bar = tqdm(train_data_loader)
    for i, (x, y) in enumerate(train_data_loader):
    # for x, y in progress_bar:
        optimizer.zero_grad()
        batch_size = y.size(0)
        reconstruction, representation, attention, probs, out = model(x)

        negatives = random_train_sample(batch_size * 20)
        with torch.no_grad():
            negatives = model.aspect.weighted_avg(model.aspect.emb(negatives), negatives == PAD)[0].view(batch_size, 20, -1)  # BME

        loss_1 = margin_loss(reconstruction, representation, negatives)
        # loss_1 = torch.zeros(1).to(device)
        loss_2 = reg_loss(model.aspect.T)
        # loss_2 = torch.zeros(1).to(device)
        # loss_3 = c_loss(out, y)
        loss_3 = torch.zeros(1).to(device)

        loss = loss_1 + loss_2 + loss_3

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            accu = (out.argmax(1) == y).float().sum().item()

        loss_1_total += loss_1.item()
        loss_2_total += loss_2.item()
        loss_3_total += loss_3.item()
        accu_total += accu
        total += batch_size

    model.eval()

    accu_total_test = 0
    total_test = 0
    # progress_bar = tqdm(test_data_loader)
    for i, (x, y) in enumerate(test_data_loader):
    # for x, y in progress_bar:

        batch_size = y.size(0)
        reconstruction, representation, attention, probs, out = model(x)

        with torch.no_grad():
            accu = (out.argmax(1) == y).float().sum().item()

        accu_total_test += accu
        total_test += batch_size

    metrics = (

        loss_1_total / total,
        loss_2_total / total,
        loss_3_total / total,
        accu_total / total,
        accu_total_test / total_test,
    )

    progress_bar.set_description("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(*metrics))

    metrics_history.append(metrics)

print(np.array(metrics_history).max(0)[-2:])



