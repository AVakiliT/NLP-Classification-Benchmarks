import math
import os
import pickle
import random
from collections import defaultdict, Counter
from itertools import chain, combinations
from typing import Any

import pandas as pd
from colorama import Fore
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
from torch.utils.data import Dataset, DataLoader
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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
#
# DATASET = 'yelp_full'
# MAX_LEN = 200
# N_EPOCHS = 6
# NUM_CLASSES = 5

DATASET = 'ng20'
MAX_LEN = 200
N_EPOCHS = 15
NUM_CLASSES = 20

BATCH_SIZE = 32
LR = 1e-3
MIN_FREQ = 4
EMBEDDING_DIM = 100
EPSILON = 1e-13
INF = 1e13
HIDDEN_DIM = 100
PAD_FIRST = False
FREEZE_EMB = False
USE_EMBED=True
# TRUNCATE_FIRST = False
# SORT_BATCHES = False

# print('Dataset ' + DATASET + ' loaded.')

# rem = stop_words.STOP_WORDS.union({'.', ',', '"', ':', ';', '-'})
rem = {}

# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%

tokenize = lambda x: x.split()

special_tokens = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']
UNK, PAD, SOS, EOS = 0, 1, 2, 3


def read_train(fname):
    df = pd.read_csv(fname, sep=',')
    texts_counter = Counter(chain.from_iterable(map(tokenize, df.text)))
    unk_count = 0
    for w in list(texts_counter.keys()):
        f = texts_counter[w]
        if f < MIN_FREQ:
            unk_count += f
            texts_counter.pop(w)
    words_set = set(texts_counter.keys())
    itos = ['<UNK>', '<PAD>', '<SOS>', '<EOS>'] + list(words_set)
    stoi = {v: i for i, v in enumerate(itos)}
    texts_idx = df.text.apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x.split()][:MAX_LEN]))
    texts_idx_sos = df.text.apply(lambda x: ([SOS] + [stoi[i] if i in stoi else UNK for i in x.split()][:MAX_LEN]))
    texts_idx_eos = df.text.apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x.split()][:MAX_LEN] + [EOS]))
    labels = df.label
    return stoi, itos, texts_idx, texts_idx_sos, texts_idx_eos, labels


def read_eval(fname, stoi):
    df = pd.read_csv(fname, sep=',')
    texts_idx = df.text.apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x.split()][:MAX_LEN]))
    texts_idx_sos = df.text.apply(lambda x: ([SOS] + [stoi[i] if i in stoi else UNK for i in x.split()][:MAX_LEN]))
    texts_idx_eos = df.text.apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x.split()][:MAX_LEN] + [EOS]))
    labels = df.label
    return texts_idx, texts_idx_sos, texts_idx_eos, labels


class ClassificationDataset(Dataset):

    def __init__(self, texts_idx, texts_idx_sos, texts_idx_eos, labels) -> None:
        super().__init__()
        self.texts_eos = texts_idx_eos
        self.texts_sos = texts_idx_sos
        self.texts = texts_idx
        self.labels = labels

    def __getitem__(self, index: int):
        return self.texts[index], self.texts_sos[index], self.texts_eos[index], self.labels[index]

    def __len__(self) -> int:
        return self.labels.__len__()


stoi, itos, train_text_idx, train_texts_idx_sos, train_texts_idx_eos, train_labels = read_train(
    DATASET + '/train_clean.csv')
train_dataset = ClassificationDataset(train_text_idx, train_texts_idx_sos, train_texts_idx_eos, train_labels)
test_dataset = ClassificationDataset(*read_eval(DATASET + '/test_clean.csv', stoi))


def pad(s, l):
    if PAD_FIRST:
        return [PAD] * (l - len(s)) + s
    else:
        return s + [PAD] * (l - len(s))


def collate(batch):
    # m = max([len(i[0]) for i in batch])
    m = MAX_LEN
    texts = torch.LongTensor([pad(item[0], m) for item in batch])
    texts_sos = torch.LongTensor([pad(item[1], m + 1) for item in batch])
    texts_eos = torch.LongTensor([pad(item[2], m + 1) for item in batch])
    labels = torch.LongTensor([item[3] for item in batch])
    return texts, texts_sos, texts_eos, labels


train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)

VOCAB_LEN = len(itos)
# %%


# TEXT = Field(sequential=True, use_vocab=True, fix_length=MAX_LEN,
#              tokenize=lambda x: [i for i in x.split() if i not in rem],
#              # include_lengths=True,
#              batch_first=True, pad_first=PAD_FIRST, truncate_first=TRUNCATE_FIRST)
#
# LABEL = Field(sequential=False, use_vocab=False, batch_first=True)
#
# columns = [('text', TEXT),
#            ('label', LABEL)]
#
# train = TabularDataset(
#     path=DATASET + '/train_clean.csv',
#     format='csv',
#     fields=columns,
#     skip_header=True
# )
#
# test = TabularDataset(
#     path=DATASET + '/test_clean.csv',
#     format='csv',
#     fields=columns,
#     skip_header=True
# )
#
# TEXT.build_vocab(train, min_freq=MIN_FREQ)
#
# PAD = 1
# # %%
# train_iter = BucketIterator(
#     train,
#     BATCH_SIZE,
#     device=device,
#     repeat=False,
#     shuffle=True,
#     sort=SORT_BATCHES,
#     sort_within_batch=True,
#     sort_key=lambda x: len(x.text),
# )
#
# test_iter = BucketIterator(
#     test,
#     BATCH_SIZE,
#     device=device,
#     repeat=False,
#     shuffle=False,
#     sort=True,
#     sort_within_batch=True,
#     sort_key=lambda x: len(x.text),
# )
#
#
# class TrainIterWrap:
#
#     def __init__(self, iterator) -> None:
#         super().__init__()
#         self.iterator = iterator
#
#     def __iter__(self):
#         for batch in self.iterator:
#             yield batch.text, batch.label
#
#     def __len__(self):
#         return len(self.iterator)
#
#
# train_data_loader = TrainIterWrap(train_iter)
# test_data_loader = TrainIterWrap(test_iter)

# %%
if USE_EMBED:
    print('Reading Embeddings...')
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        '/home/amir/IIS/Datasets/embeddings/glove.6B.' + str(EMBEDDING_DIM)
        + 'd.txt.w2vformat',
        binary=True)

    embedding_weights = torch.zeros(VOCAB_LEN, EMBEDDING_DIM)
    nn.init.normal_(embedding_weights)

    unmatch = []
    for i, word in enumerate(itos):
        if word in w2v and i != PAD:
            embedding_weights[i] = torch.Tensor(w2v[word])
        else:
            unmatch.append(word)
            if i == PAD:
                embedding_weights[i] = torch.zeros(EMBEDDING_DIM)

    print(len(unmatch) * 100 / VOCAB_LEN, '% of embeddings didn\'t match')

    embedding_weights.to(device)


def get_emb():
    if USE_EMBED:
        emb = nn.Embedding(VOCAB_LEN, EMBEDDING_DIM, padding_idx=PAD, _weight=embedding_weights.clone())
        emb.requires_grad = not FREEZE_EMB
    else:
        emb = nn.Embedding(VOCAB_LEN, EMBEDDING_DIM, padding_idx=PAD)
    return emb


# %%
train_list = list(train_data_loader)
all_train_text = torch.cat([x[0] for x in train_list], 0)
del train_list


def random_train_sample(n):
    return all_train_text[np.random.randint(0, len(all_train_text), n)]


# %%

def sentence_emb_avg(xx, mask):
    # return xx.max(1)[0]
    len_s = (mask ^ 1).long().sum(1, keepdim=True).float()
    y_s = xx.sum(1) / len_s
    return y_s


# class LSTM_LM(nn.Module):
#     def __init__(self, latent_size=50, hidden_size=512):
#         super().__init__()
#         self.latent_size = latent_size
#         self.emb = get_emb()
#         self.encoder_rnn = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=hidden_size,
#                                    bidirectional=True, batch_first=True, num_layers=1)
#         self.decoder_rnn = nn.LSTM(input_size=EMBEDDING_DIM + latent_size, hidden_size=hidden_size * 2,
#                                    batch_first=True, num_layers=1)
#
#         self.output2vocab = nn.Linear(hidden_size * 2, VOCAB_LEN)
#         self.fc = nn.Linear(hidden_size * 2, latent_size)
#
#     def encode(self, xx1):
#         states, _ = self.encoder_rnn(xx1)
#         z = self.fc(states[:, -1, :]).relu()  # BZ
#         return torch.zeros(1).to(device), z
#
#     def decode(self, xx2, z):
#         xx = torch.cat([xx2, z.unsqueeze(1).repeat(1, xx2.shape[1], 1)], -1)
#
#         xx, _ = self.decoder_rnn(xx)
#         xx = self.output2vocab(xx)  # BSV
#
#         return xx
#
#     def forward(self, encoder_input, decoder_input):
#         xx1 = self.emb(encoder_input)
#         kld, z = self.encode(xx1)
#
#         xx2 = self.emb(decoder_input)
#         logits = self.decode(xx2, z)
#
#         return logits, kld


class LSTM_LM_Bowman(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.emb = get_emb()
        self.encoder_rnn = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=hidden_size,
                                   bidirectional=True, batch_first=True, num_layers=1)
        self.decoder_rnn = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=hidden_size * 2,
                                   batch_first=True, num_layers=1)

        self.output2vocab = nn.Linear(hidden_size * 2, VOCAB_LEN)
        self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)

    def encode(self, xx1):
        states, _ = self.encoder_rnn(xx1)
        z = self.fc(states[:, -1, :]).relu()  # BZ
        return torch.zeros(1).to(device), z

    def decode(self, xx2, z):
        # xx = torch.cat([xx2, z.unsqueeze(1).repeat(1, xx2.shape[1], 1)], -1)

        xx, _ = self.decoder_rnn(xx2, z)
        xx = self.output2vocab(xx)  # BSV

        return xx

    def forward(self, encoder_input, decoder_input):
        xx1 = self.emb(encoder_input)
        kld, z = self.encode(xx1)

        xx2 = self.emb(decoder_input)
        logits = self.decode(xx2, (z.unsqueeze(0), torch.zeros_like(z.unsqueeze(0))))

        return logits, kld

class LSTM_VAE(nn.Module):
    def __init__(self, latent_size=50, hidden_size=512):
        super().__init__()
        self.latent_size = latent_size
        self.emb = get_emb()
        self.encoder_rnn = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=hidden_size,
                                   bidirectional=True, batch_first=True, num_layers=1)
        self.decoder_rnn = nn.LSTM(input_size=EMBEDDING_DIM + latent_size, hidden_size=hidden_size * 2,
                                   batch_first=True, num_layers=1)

        self.output2vocab = nn.Linear(hidden_size * 2, VOCAB_LEN)
        self.fc = nn.Linear(latent_size, latent_size)

        self.context2mu = nn.Linear(hidden_size * 2, latent_size)
        self.context2logvar = nn.Linear(hidden_size * 2, latent_size)

    def encode(self, xx1):
        states, _ = self.encoder_rnn(xx1)
        z, kld = self.reparametrize(states[:, -1, :])  # BZ
        return kld, z

    def reparametrize(self, context):
        mu = self.context2mu(context)
        logvar = self.context2logvar(context)
        std = logvar.mul(.5).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        # kld = (logvar - mu.pow(2) - logvar.exp() + 1).sum(-1).mul(-0.5).sum()
        kld = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, -1)
        kld = kld.sum()
        return z, kld

    def decode(self, xx2, z):
        z_ = self.fc(z).relu()
        xx = torch.cat([xx2, z_.unsqueeze(1).repeat(1, xx2.shape[1], 1)], -1)

        xx, _ = self.decoder_rnn(xx)
        xx = self.output2vocab(xx)  # BSV

        return xx

    def forward(self, encoder_input, decoder_input):
        xx1 = self.emb(encoder_input)
        kld, z = self.encode(xx1)

        xx2 = self.emb(decoder_input)
        logits = self.decode(xx2, z)

        return logits, kld

    def sample(self, zz=None, seq_len=MAX_LEN):
        with torch.no_grad():
            if zz is None:
                zz = torch.randn(self.latent_size)

            z = torch.Tensor(zz).to(device).unsqueeze(0)

            decoder_input = torch.LongTensor([SOS]).unsqueeze(0)

            sentence = ''

            for i in range(seq_len):
                xx2 = self.emb(decoder_input.to(device))
                logits = self.decode(xx2, z)
                logits = logits[0, -1].softmax(0)
                next_word_id = np.random.choice(range(VOCAB_LEN), p=logits.detach().cpu().numpy())

                if next_word_id == EOS:
                    break

                sentence += ' ' + itos[next_word_id]

                decoder_input = torch.cat([decoder_input, torch.LongTensor([next_word_id]).unsqueeze(0)], 1)

        return sentence.strip()


class LSTM_VAE_CNN_AE(nn.Module):

    def __init__(self, latent_size=32, hidden_size=512, num_dilation_layers=5):
        super().__init__()

        self.latent_size = latent_size
        channel_external_size = hidden_size * 2
        channel_internal_size = hidden_size

        self.emb = get_emb()
        self.encoder_rnn = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=hidden_size,
                                   bidirectional=True, batch_first=True, num_layers=1)
        self.context2mu = nn.Linear(hidden_size * 2, latent_size)
        self.context2logvar = nn.Linear(hidden_size * 2, latent_size)

        self.latent2channel = nn.Linear(latent_size, channel_external_size - EMBEDDING_DIM)

        conv_params = [
            [
                (channel_external_size, channel_internal_size, 1),
                (channel_internal_size, channel_internal_size, 3, 1, 2 ** (i + 1), 2 ** i),
                (channel_internal_size, channel_external_size, 1)
            ] for i in range(num_dilation_layers)
        ]

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(*p1),
                nn.ReLU(),
                nn.Dropout(.1),
                nn.Conv1d(*p2),
                nn.ReLU(),
                nn.Dropout(.1),
                nn.Conv1d(*p3),
                nn.ReLU(),
                nn.Dropout(.1),
            ) for p1, p2, p3 in conv_params
        ])

        self.output2vocab = nn.Linear(channel_external_size, VOCAB_LEN)

    def reparametrize(self, context):
        mu = self.context2mu(context)
        logvar = self.context2logvar(context)
        std = logvar.mul(.5).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        # kld = (logvar - mu.pow(2) - logvar.exp() + 1).sum(-1).mul(-0.5).sum()
        kld = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, -1)
        kld = kld.sum()
        return z, kld

    def forward(self, encoder_input, decoder_input):

        xx1 = self.emb(encoder_input)
        kld, z = self.encode(xx1)

        xx2 = self.emb(decoder_input)
        logits = self.decode(xx2, z)

        return logits, kld

    def encode(self, xx1):
        states, _ = self.encoder_rnn(xx1)
        z, kld = self.reparametrize(states[:, -1, :])  # BZ
        return kld, z

    def decode(self, xx2, z):
        z_ = self.latent2channel(z).relu()
        xx = torch.cat([xx2, z_.unsqueeze(1).repeat(1, xx2.shape[1], 1)], -1)
        xx = xx.transpose(1, 2)

        res = xx
        for conv in self.convs:
            xx = conv(xx)

            xx = xx[:, :, :(xx.shape[2] - conv[3].padding[0])].contiguous()  # shift by 1

            xx = xx.add(res).relu()
            res = xx

        xx = xx.transpose(1, 2)  # BSH
        xx = self.output2vocab(xx)  # BSV
        return xx

    def sample(self, zz=None, seq_len=MAX_LEN):
        with torch.no_grad():
            if zz is None:
                zz = torch.randn(self.latent_size)

            z = torch.Tensor(zz).to(device).unsqueeze(0)

            decoder_input = torch.LongTensor([SOS]).unsqueeze(0)

            sentence = ''

            for i in range(seq_len):
                xx2 = self.emb(decoder_input.to(device))
                logits = self.decode(xx2, z)
                logits = logits[0, -1].softmax(0)
                next_word_id = np.random.choice(range(VOCAB_LEN), p=logits.detach().cpu().numpy())

                if next_word_id == EOS:
                    break

                sentence += ' ' + itos[next_word_id]

                decoder_input = torch.cat([decoder_input, torch.LongTensor([next_word_id]).unsqueeze(0)], 1)

        return sentence.strip()

    def get_z(self, x):
        with torch.no_grad():
            x = self.emb(x)
            z, _ = self.encode(x)
            return z


class ConvDeconv(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.emb = get_emb()
        self.encoders = nn.ModuleList([
            nn.Conv1d(EMBEDDING_DIM, 300, 5, 2),
            nn.Conv1d(300, 600, 5, 2),
            nn.Conv1d(600, 500, 5, 2),
        ])
        self.decoders = nn.ModuleList([
            nn.ConvTranspose1d(500, 600, 5, 2),
            nn.ConvTranspose1d(600, 300, 5, 2),
            nn.ConvTranspose1d(300, EMBEDDING_DIM, 5, 2),
        ])
        self.context2mu = nn.Linear(2000, 2000)
        self.context2logvar = nn.Linear(2000, 2000)

    def encoder(self, xx):
        shapes = [xx.shape]
        for conv in self.encoders:
            xx = conv(xx).relu()
            shapes.append(xx.shape)
        return xx, shapes[:-1]

    def decoder(self, z, shapes):
        xx = z
        for deconv, shape in zip(self.decoders, shapes):
            xx = deconv(xx, output_size=shape).relu()
        return xx

    def reparametrize(self, context):
        mu = self.context2mu(context)
        logvar = self.context2logvar(context)
        std = logvar.mul(.5).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        # kld = (logvar - mu.pow(2) - logvar.exp() + 1).sum(-1).mul(-0.5).sum()
        kld = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, -1)
        kld = kld.sum()
        return z, kld

    def forward(self, x, _):
        xx = self.emb(x)
        xx, shapes = self.encoder(xx.transpose(1, 2))
        xx = xx.transpose(1, 2)

        b, s, _ = xx.shape
        xx = xx.reshape(b, -1)
        z, kld = self.reparametrize(xx)
        z = z.reshape(b, s, -1)

        xx = self.decoder(z.transpose(1, 2), shapes[::-1]).transpose(1, 2)

        sims = torch.einsum('bse,ve->bsv', [xx, self.emb.weight])
        sims_ = torch.einsum('bs,v->bsv', [xx.norm(dim=-1), self.emb.weight.norm(dim=-1)])
        sims_[sims_ == 0] = INF
        sims = sims / sims_

        return sims, kld


# def margin_loss(reconstruction, sentence_weighted_average, negatives=None):
#     """
#     :param reconstruction: BE
#     :param sentence_weighted_average: BE
#     :param negatives: BME
#     :return: j: B
#     """
#
#     if negatives is not None:
#         m = negatives.shape[1]
#         # r_sim = torch.einsum('be,be->b', [reconstruction, representation]).repeat(m, 1)
#         # n_sim = torch.einsum('be,bme->mb', [reconstruction, negatives])
#
#         r_sim_ = F.cosine_similarity(reconstruction, sentence_weighted_average, dim=-1).repeat(m, 1)
#         n_sim_ = F.cosine_similarity(reconstruction.unsqueeze(1).repeat(1, m, 1), negatives, dim=-1).t()
#
#         # r_sim_ = (reconstruction.norm(dim=-1) * representation).repeat(m, 1)
#         # n_sim_ = torch.einsum('b,bm->mb', [reconstruction.nomr(dim=-1), negatives.norm(dim=-1)])
#     else:
#         r_sim = torch.einsum('be,be->b', [reconstruction, sentence_weighted_average])
#         n_sim = 0
#
#     # j = F.relu(1 - r_sim + n_sim).sum()
#     j = F.relu(1 - r_sim_ + n_sim_).sum()
#
#     if negatives is not None:
#         j = j / m
#
#     return j
#
#
# def vae_loss(recon_x, x, mu, logvar):
#     MSE = F.mse_loss(recon_x, x, reduction='sum')
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#
#     return MSE + KLD, MSE, KLD
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


# def insert_sos(xx):
#     if not PAD_FIRST:
#         torch.cat([torch.ones(xx.shape[0], dtype=torch.long).mul(SOS).unsqueeze(1), x], 1)
#     else:
#         raise NotImplementedError
#
# def insert_eos(xx):
#     if not PAD_FIRST:
#         torch.cat([torch.ones(xx.shape[0], dtype=torch.long).mul(SOS).unsqueeze(1), x], 1)
#     else:
#         raise NotImplementedError

def perplexity(logits, target):
    with torch.no_grad():
        ppl = logits.log_softmax(-1).gather(2, target.unsqueeze(2)).neg().mean(1).exp().squeeze(-1)
        return ppl


# %%

model = LSTM_VAE_CNN_AE(latent_size=32, hidden_size=128)
# model = LSTM_LM(latent_size=32, hidden_size=128)
# model = LSTM_VAE(latent_size=32, hidden_size=128)
# model = ConvDeconv()
# model = LSTM_LM_Bowman(hidden_size=128)

model = model.to(device)

c_loss = nn.CrossEntropyLoss(reduction='sum')

metrics_history_all = []

optimizer = torch.optim.Adam(model.parameters(), betas=(0.5, .999))
metrics_history = []
# progress_bar = tqdm(range(1, N_EPOCHS + 1))
# for i_epoch in progress_bar:
for i_epoch in range(1, N_EPOCHS):
    model.train()
    rec_loss_total = 0
    kld_loss_total = 0
    loss_total = 0
    ppl_total = 0
    total = 0
    progress_bar = tqdm(train_data_loader)
    for i, (x, x_sos, x_eos, y) in enumerate(progress_bar):
        x = x.to(device)  # encoder_inpuy
        x_sos = x_sos.to(device)  # decoder input
        x_eos = x_eos.to(device)  # decoder output
        y = y.to(device)
        batch_size = y.size(0)

        target = x_eos

        optimizer.zero_grad()

        logits, kld = model(x, x_sos)

        cross_entropy = F.cross_entropy(logits.view(-1, VOCAB_LEN), target.view(-1), reduction='sum')

        loss = cross_entropy + kld * (.01 + (.99 * ((total / len(train_dataset)) / 10 + (i_epoch / min(N_EPOCHS, 10)))))
        loss.backward()
        optimizer.step()

        ppl = perplexity(logits, target).sum()

        rec_loss_total += cross_entropy.item()
        kld_loss_total += kld.item()
        loss_total += loss.item()
        ppl_total += ppl.item()
        total += batch_size

        metrics = (
            rec_loss_total / total,
            kld_loss_total / total,
            loss_total / total,
            ppl_total / total
        )

        progress_bar.set_description(
            Fore.RESET + '[ EPOCH: {:02d} ][ TRN NLL: {:.3f} KLD: {:.3f} LSS: {:.3f} PPL: {:.3f}]'.format(i_epoch,
                                                                                                          *metrics))

    with torch.no_grad():
        model.eval()
        rec_loss_total_test = 0
        kld_loss_total_test = 0
        # loss_total_test = 0
        ppl_total_test = 0
        total_test = 0
        progress_bar = tqdm(test_data_loader)
        for i, (x, x_sos, x_eos, y) in enumerate(progress_bar):
            x = x.to(device)  # encoder_inpuy
            x_sos = x_sos.to(device)  # decoder input
            x_eos = x_eos.to(device)  # decoder output
            y = y.to(device)
            batch_size = y.size(0)

            target = x_eos

            logits, kld = model(x, x_sos)

            cross_entropy = F.cross_entropy(logits.view(-1, VOCAB_LEN), target.view(-1), reduction='sum')

            ppl = perplexity(logits, target).sum()

            rec_loss_total_test += cross_entropy.item()
            kld_loss_total_test += kld.item()
            # loss_total_test += loss.item()
            ppl_total_test += ppl.item()
            total_test += batch_size

            metrics = (
                rec_loss_total_test / total_test,
                kld_loss_total_test / total_test,
                # loss_total_test / total_test,
                ppl_total_test / total_test
            )

            progress_bar.set_description(
                Fore.YELLOW + '[ EPOCH: {:02d} ][ TST NLL: {:.3f} KLD: {:.3f} PPL: {:.3f}]'.format(i_epoch, *metrics))
    #
    # metrics = (
    #     loss_0_total / total,
    #     loss_1_total / total,
    #     loss_2_total / total,
    #     loss_3_total / total,
    #     # accu_total / total,
    #     # accu_total_test / total_test,
    # )

    # progress_bar.set_description("{:.3f} {:.3f} {:.3f} {:.3f}".format(*metrics))
    #
    # metrics_history.append(metrics)

# %%
# def topic_words(n_top_words=10):
#     sims = (model.aspect.T.detach() @ model.aspect.emb.weight.t().detach()) / (
#             model.aspect.T.detach().norm(dim=-1, keepdim=True).detach() @ model.aspect.emb.weight.detach().norm(dim=-1,
#                                                                                                                 keepdim=True).t())
#     sims[torch.isnan(sims)] = -1
#
#     sims = sims.cpu()
#     top_words = sims.sort(dim=-1, descending=True)[1][:, :n_top_words]
#     for k, beta_k in enumerate(top_words):
#         topic_words = [itos[w_id.item()] for w_id in beta_k]
#         print('Topic {}: {}'.format(k, ' '.join(topic_words)))
#     return top_words
#
#
# top_words = topic_words(15)
#
# # %%
#
# inverse_index = defaultdict(list)
# for word in tqdm(range(VOCAB_LEN)):
#     for did, doc in enumerate(train_dataset.texts):
#         if word in doc:
#             inverse_index[word].append(did)
#
#
# # %%
#
#
# def co_document_frequency(w1, w2):
#     return len(set(inverse_index[w1]).intersection(set(inverse_index[w2])))
#
#
# def document_frequency(w1):
#     return len(inverse_index[w1])
#
#
# all_scores = []
# for topic in tqdm(top_words.numpy()):
#     score = 0
#     for w1, w2 in combinations(topic, 2):
#         score += np.log((co_document_frequency(w1, w2) + 1) / (document_frequency(w1) + document_frequency(w1) + 1))
#     all_scores.append(score)
# all_scores = np.array(all_scores)
# print(all_scores)
# print(all_scores.mean())
