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
from texttable import Texttable
from torch import nn
from torch.nn import functional as F, Parameter, init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm

SEED = 2

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

DATASET = 'yelp_full'
MAX_LEN = 200
N_EPOCHS = 4
NUM_CLASSES = 5

# DATASET = 'ng20'
# MAX_LEN = 200
# N_EPOCHS = 100
# NUM_CLASSES = 20

BATCH_SIZE = 8
LR = 1e-3
MIN_FREQ = 64
EMBEDDING_DIM = 100
EPSILON = 1e-13
INF = 1e13
HIDDEN_DIM = 100
PAD_FIRST = True
TRUNCATE_FIRST = False
SORT_BATCHES = False

print('Dataset ' + DATASET + ' loaded.')

# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TEXT = Field(sequential=True, use_vocab=True, fix_length=MAX_LEN, tokenize=lambda x: x.split(),
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
# train_list = list(train_data_loader)
# all_train_text = torch.cat([x[0] for x in train_list], 0)
# del train_list
#
#
# def random_train_sample(n):
#     return all_train_text[np.random.randint(0, len(all_train_text), n)]


# %%

# def sentence_emb_avg(xx, mask):
#     len_s = (mask ^ 1).long().sum(1, keepdim=True).float()
#     y_s = xx.sum(1) / len_s
#     return y_s


# class Aspect(nn.Module):
#     def __init__(self, emb_dim, asp_dim, inf):
#         super().__init__()
#         self.emb = get_emb()
#         self.asp_dim = asp_dim
#         self.emb_dim = emb_dim
#         self.inf = inf
#         # self.M = nn.Parameter(torch.randn(emb_dim, emb_dim))
#         self.S = nn.Parameter(torch.randn(emb_dim))
#         self.W = nn.Linear(emb_dim, asp_dim)
#         self.T = nn.Parameter(torch.randn(asp_dim, emb_dim))  #
#
#         self.reset_params()
#
#     def reset_params(self):
#         bound = 1 / math.sqrt(self.emb_dim)
#         # init.uniform_(self.M, -bound, bound)
#         init.uniform_(self.S, -bound, bound)
#
#         X = self.emb.weight.detach().cpu().numpy()
#         kc = KMeans(n_jobs=8, n_clusters=self.asp_dim)
#         kc_fit = kc.fit(X)
#         centroids = kc_fit.cluster_centers_
#         self.T = nn.Parameter(torch.FloatTensor(centroids))
#
#     def encode(self, xx, mask):
#         """
#
#         :param xx: BSE
#         :param mask: BS
#         :return: BT
#         """
#
#         # y_s = sentence_emb_avg(xx, mask)
#
#         z_s = self.weighted_avg(xx, mask)
#
#         p_t = F.relu(self.W(z_s))
#
#
#         return p_t, z_s
#
#     def weighted_avg(self, xx, mask):
#         # d_i = torch.einsum('bse,e->bs', [xx, self.S])
#         # d_i.masked_fill_(mask, -INF)
#         # a_i = F.softmax(d_i, -1)
#         # z_s = torch.einsum('bse,bs->be', [xx, a_i])
#         # return z_s
#         len_s = (mask ^ 1).long().sum(1, keepdim=True).float()
#         y_s = xx.sum(1) / len_s
#         return y_s
#
#
#     def decode(self, p_t):
#         r_s = p_t @ self.T
#         return r_s
#
#     def forward(self, x):
#         mask = x == PAD
#         x = self.emb(x)
#         p_t, z_s = self.encode(x, mask)
#         r_s = self.decode(p_t)
#         return p_t, z_s, r_s
#
#
# class Classifier(nn.Module):
#
#     def __init__(self, aspect_dim):
#         super().__init__()
#         self.aspect_dim = aspect_dim
#         self.final = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(aspect_dim, NUM_CLASSES)
#         )
#
#     def forward(self, p_t):
#         return self.final(p_t)
#
#
# class Model(nn.Module):
#
#     def __init__(self, aspects):
#         super().__init__()
#         self.aspect = Aspect(EMBEDDING_DIM, aspects, INF)
#         self.classifier = Classifier(aspects)
#
#     def forward(self, x):
#         p_t, z_s, r_s = self.aspect(x)
#         out = self.classifier(p_t)
#         return out, r_s, z_s
#
#
# def margin_loss(reconstruction, representation, negatives=None):
#     """
#     :param reconstruction: BE
#     :param representation: BE
#     :param negatives: BME
#     :return: j: B
#     """
#
#     if negatives is not None:
#         m = negatives.shape[1]
#         r_sim = torch.einsum('be,be->b', [reconstruction, representation]).repeat(m, 1)
#         n_sim = torch.einsum('be,bme->mb', [reconstruction, negatives])
#
#         # r_sim_ = (reconstruction.norm(dim=-1) * representation).repeat(m, 1)
#         # n_sim_ = torch.einsum('b,bm->mb', [reconstruction.nomr(dim=-1), negatives.norm(dim=-1)])
#     else:
#         r_sim = torch.einsum('be,be->b', [reconstruction, representation])
#         n_sim = 0
#
#     j = F.relu(1 - r_sim + n_sim).sum()
#
#     if negatives is not None:
#         j = j / m
#
#     return j
#
#
# def reg_loss(t):
#     """
#
#     :param t: ET
#     :return: 1
#     """
#     t_dot_tt = (t @ t.t()) / (t.norm(dim=-1, keepdim=True) @ t.norm(dim=-1, keepdim=True).t())
#     t_dot_tt = t_dot_tt - torch.eye(t_dot_tt.shape[0]).to(device)
#     u = t_dot_tt.norm()
#     return u


# %%
class ConvolutionEncoder(nn.Module):
    def __init__(self, embedding, filter_size, filter_shape, latent_size):
        super(ConvolutionEncoder, self).__init__()
        self.embed = embedding
        self.convs1 = nn.Conv1d(EMBEDDING_DIM, filter_size, filter_shape, stride=2)
        self.bn1 = nn.BatchNorm1d(filter_size)
        self.convs2 = nn.Conv1d(filter_size, filter_size * 2, filter_shape, stride=2)
        self.bn2 = nn.BatchNorm1d(filter_size * 2)

        self.len_1 = (MAX_LEN - (filter_shape - 1)) // 2  # 28
        self.len_2 = (self.len_1 - (filter_shape - 1)) // 2

        self.convs3 = nn.Conv1d(filter_size * 2, latent_size, self.len_2, stride=1)

        # weight initialize for conv layer
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.embed(x)

        # reshape for convolution layer
        x = x.transpose(1, 2)

        h1 = F.relu(self.bn1(self.convs1(x)))
        h2 = F.relu(self.bn2(self.convs2(h1)))
        h3 = F.relu(self.convs3(h2))

        return h3


class DeconvolutionDecoder(nn.Module):
    def __init__(self, embedding, latent_size, len_1, len_2, filter_size, filter_shape=5, tau=.01):
        super(DeconvolutionDecoder, self).__init__()
        self.tau = tau
        self.embed = embedding

        self.shape_1 = [filter_size * 2, len_2]
        self.shape_2 = [filter_size, len_1]
        self.shape_3 = [EMBEDDING_DIM, MAX_LEN]

        self.deconvs1 = nn.ConvTranspose1d(latent_size, filter_size * 2, kernel_size=len_2, stride=1)
        self.bn1 = nn.BatchNorm1d(filter_size * 2)
        self.deconvs2 = nn.ConvTranspose1d(filter_size * 2, filter_size, kernel_size=filter_shape, stride=2)
        self.bn2 = nn.BatchNorm1d(filter_size)
        self.deconvs3 = nn.ConvTranspose1d(filter_size, EMBEDDING_DIM, kernel_size=filter_shape, stride=2)

        # weight initialize for conv_transpose layer
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, h3):
        h2 = F.relu(self.bn1(self.deconvs1(h3, output_size=[h3.shape[0], *self.shape_1])))
        h1 = F.relu(self.bn2(self.deconvs2(h2, output_size=[h3.shape[0], *self.shape_2])))
        x_hat = F.relu(self.deconvs3(h1, output_size=[h3.shape[0], *self.shape_3]))
        x_hat = x_hat.transpose(1, 2)

        norm_e = self.embed.weight.data
        norm_e[PAD].fill_(1e-13)

        sims = torch.einsum('bse,ve->bsv', x_hat, norm_e)
        sims_norm = torch.einsum('bs,v->bsv', x_hat.norm(dim=-1), norm_e.norm(dim=-1))
        sims = sims / sims_norm
        sims = sims / self.tau
        sims = sims.transpose(1, 2)

        return sims


class AutoEncoder(nn.Module):

    def __init__(self, filter_size=100, filter_shape=5, latent_size=200):
        super().__init__()
        self.emb = get_emb()
        self.encoder = ConvolutionEncoder(self.emb, filter_size, filter_shape, latent_size)
        self.decoder = DeconvolutionDecoder(self.emb, latent_size, self.encoder.len_1, self.encoder.len_2, filter_size,
                                            filter_shape)

        vs = len(TEXT.vocab)
        self.softmax = nn.AdaptiveLogSoftmaxWithLoss(EMBEDDING_DIM, vs, cutoffs=[round(vs / 15), 3 * round(vs / 15)],
                                                     div_value=4)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruct = self.decoder(latent)
        # loss = self.softmax(reconstruct, x)
        return reconstruct


# %%
model = AutoEncoder()

model = model.to(device)

cross_loss = nn.CrossEntropyLoss(reduction='sum')

metrics_history_all = []

optimizer = torch.optim.Adam(model.parameters())
metrics_history = []
progress_bar = tqdm(range(1, N_EPOCHS + 1))
for i_epoch in progress_bar:
    model.train()
    loss_total = 0
    accu_total = 0
    total = 0
    progress_bar = tqdm(train_data_loader)
    # for i, (x, y) in enumerate(train_data_loader):
    for x, y in progress_bar:
        optimizer.zero_grad()
        batch_size = y.size(0)
        reconstruction = model(x)

        # if i % 1 == 0:
        #     print(i / len(train_data_loader))

        loss = cross_loss(reconstruction, x) / x.shape[1]

        # with torch.no_grad():
        #     negatives = random_train_sample(batch_size * 20)
        #     negatives = model.aspect.weighted_avg(model.aspect.emb(negatives), negatives == PAD).view(batch_size, 20,
        #                                                                                               -1)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = (reconstruction.argmax(1) == x).float().mean(-1).sum().item()

        loss_total += loss.item()
        accu_total += acc
        total += batch_size

        metrics = (
            loss_total / total,
            accu_total / total
        )
        progress_bar.set_description(
            "[ TRAIN LSS: {:.3f} ACC: {:.3f} ]".format(*metrics)
        )

    model.eval()
    loss_total_test = 0
    accu_total_test = 0
    total_test = 0
    for x, y in test_data_loader:
        reconstruction = model(x)

        loss = cross_loss(reconstruction, x) / x.shape[1]

        with torch.no_grad():
            acc = (reconstruction.argmax(1) == x).float().mean(-1).sum().item()

        batch_size = y.size(0)
        loss_total_test += loss.item()
        accu_total_test += acc
        total_test += batch_size

        metrics = (
            loss_total_test / total_test,
            accu_total_test / total_test
        )
        progress_bar.set_description(
            "[  TST  LSS: {:.3f} ACC: {:.3f} ]".format(*metrics)
        )

    #
    # metrics = (
    #     loss_total / total,
    #     accu_total / total,
    #     loss_total_test / total_test,
    #     accu_total_test / total_test
    # )
    # progress_bar.set_description(
    #     "[ TRAIN LSS: {:.3f} ACC: {:.3f} ][ TEST LSS: {:.3f} ACC: {:.3f} ]".format(*metrics)
    # )
    # metrics_history.append(metrics)

# %%

baselines = dict()


def model_name(model):
    try:
        a = model.name
    except:
        a = type(model).__name__
    return a


baselines[model_name(model)] = metrics_history

metrics_history_all_t = [np.array(i).T for i in baselines.values()]
model_names = list(baselines.keys())

rows = [(n, np.round(v[1].max(), 3), np.round(v[3].max(), 3)) for n, v in zip(model_names, metrics_history_all_t)]
rows = sorted(rows, key=lambda x: x[2])
t = Texttable()
t.add_rows(rows, header=False)
t.header(('Model', 'Acc Train', 'Acc Test'))
print(t.draw())
