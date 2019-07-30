import math
import os
import pickle
import random
from collections import defaultdict, Counter
from itertools import chain, combinations

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
from torch.utils.data import Dataset, DataLoader
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
MIN_FREQ = 4
EMBEDDING_DIM = 300
EPSILON = 1e-13
INF = 1e13
HIDDEN_DIM = 100
PAD_FIRST = True
TRUNCATE_FIRST = False
SORT_BATCHES = False

# print('Dataset ' + DATASET + ' loaded.')

# rem = stop_words.STOP_WORDS.union({'.', ',', '"', ':', ';', '-'})
rem = {}

# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%

tokenize = lambda x: x.split()
UNK = 0
PAD = 1


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
    itos = ['<UNK>', '<PAD>'] + list(words_set)
    stoi = {v: i for i, v in enumerate(itos)}
    texts_idx = df.text.apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x.split()][:MAX_LEN]))
    labels = df.label
    return stoi, itos, texts_idx, labels


def read_eval(fname, stoi):
    df = pd.read_csv(fname, sep=',')
    texts_idx = df.text.apply(lambda x: ([stoi[i] if i in stoi else UNK for i in x.split()][:MAX_LEN]))
    labels = df.label
    return texts_idx, labels


class ClassificationDataset(Dataset):

    def __init__(self, texts_idx, labels) -> None:
        super().__init__()
        self.texts = texts_idx
        self.labels = labels

    def __getitem__(self, index: int):
        return self.texts[index], self.labels[index]

    def __len__(self) -> int:
        return self.labels.__len__()


stoi, itos, train_text_idx, train_labels = read_train(DATASET + '/train_clean.csv')
train_dataset = ClassificationDataset(train_text_idx, train_labels)
test_dataset = ClassificationDataset(*read_eval(DATASET + '/test_clean.csv', stoi))


def collate(batch):
    # m = max([len(i[0]) for i in batch])
    m = MAX_LEN
    texts = torch.LongTensor([[1] * (m - len(item[0])) + item[0] for item in batch])
    labels = torch.LongTensor([item[1] for item in batch])
    return [texts, labels]


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
    return nn.Embedding(VOCAB_LEN, EMBEDDING_DIM, padding_idx=PAD, _weight=embedding_weights.clone())


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


class Aspect(nn.Module):
    def __init__(self, emb_dim=EMBEDDING_DIM, hidden_dim=100, asp_dim=50):
        super().__init__()
        self.emb = get_emb()
        self.asp_dim = asp_dim
        self.emb_dim = emb_dim
        # self.inf = inf
        # self.M = nn.Parameter(torch.randn(emb_dim, emb_dim))
        # self.Source = nn.Parameter(torch.randn(emb_dim))
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, asp_dim)
        self.fc22 = nn.Linear(hidden_dim, asp_dim)

        self.fcg1 = nn.Linear(asp_dim, asp_dim)
        self.fcg2 = nn.Linear(asp_dim, asp_dim)
        self.fcg3 = nn.Linear(asp_dim, asp_dim)
        self.fcg4 = nn.Linear(asp_dim, asp_dim)

        self.T = nn.Parameter(torch.randn(asp_dim, emb_dim))  #
        # self.conv = nn.Conv1d(asp_dim, asp_dim, kernel_size=51, stride=1, padding=25)
        self.reset_params()

    def reset_params(self):
        # bound = 1 / math.sqrt(self.emb_dim)
        # init.uniform_(self.M, -bound, bound)
        # init.uniform_(self.Source, -bound, bound)

        X = self.emb.weight.detach().cpu().numpy()
        kc = KMeans(n_jobs=8, n_clusters=self.asp_dim)
        kc_fit = kc.fit(X)
        centroids = kc_fit.cluster_centers_
        self.T = nn.Parameter(torch.FloatTensor(centroids))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate(self, h):
        g1 = torch.tanh(self.fcg1(h))
        g1 = torch.tanh(self.fcg2(g1))
        g1 = torch.tanh(self.fcg3(g1))
        g1 = torch.tanh(self.fcg4(g1))
        g1 = g1.add(h)
        return g1.softmax(-1)

    def encode(self, sentence_weighted_average):
        p_t_ = self.fc1(sentence_weighted_average)  # BA
        mu = self.fc21(p_t_)  # BA
        logvar = self.fc22(p_t_)  # BA


        return mu, logvar

    def weighted_avg(self, xx, mask, y_s=None):
        d_i = torch.einsum('bse,be->bs', [xx, y_s])
        d_i.masked_fill_(mask, -INF)
        a_i = F.softmax(d_i, -1)
        z_s = torch.einsum('bse,bs->be', [xx, a_i])
        return z_s, a_i

        ####

        # g = torch.einsum('ae,bse->bas', [self.T, xx])
        # g_hat = torch.einsum('a,bs->bas', [self.T.norm(dim=-1), xx.norm(dim=-1)])
        # g_hat[g_hat == 0] = EPSILON
        # g = g / g_hat
        # u = F.relu(self.conv(g))
        # m = u.max(1)[0]  # BS
        # a_i = m.masked_fill(mask, -INF)
        # a_i = F.softmax(a_i, -1)
        # z_s = torch.einsum('bse,bs->be', [xx, a_i])
        # return z_s, a_i

    def decode(self, z):
        r_s = z @ self.T
        return r_s

    def forward(self, x):
        mask = x == PAD
        xx = self.emb(x)

        sentence_average = sentence_emb_avg(xx, mask)

        sentence_weighted_average, a_i = self.weighted_avg(xx, mask, sentence_average)

        mu, logvar = self.encode(sentence_weighted_average)
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)
        recons = self.decode(g)

        return recons, mu, logvar, z, sentence_weighted_average





# class Classifier(nn.Module):
#
#     def __init__(self, aspect_dim):
#         super().__init__()
#         self.aspect_dim = aspect_dim
#         self.final = nn.Sequential(
#             nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM // 2),
#             nn.ReLU(),
#             nn.Linear(EMBEDDING_DIM // 2, NUM_CLASSES)
#         )
#
#     def forward(self, p_t):
#         return self.final(p_t)


class AspectExtractor(nn.Module):

    def __init__(self, asp_dim=20):
        super().__init__()
        self.aspect = Aspect(asp_dim=asp_dim)
        # self.classifier = Classifier(aspects)

    def forward(self, x):
        reconstruction, mu, logvar, z, sentence_weighted_average = self.aspect(x)
        # out = self.classifier(z_s)
        return reconstruction, mu, logvar, z, sentence_weighted_average


def margin_loss(reconstruction, sentence_weighted_average, negatives=None):
    """
    :param reconstruction: BE
    :param sentence_weighted_average: BE
    :param negatives: BME
    :return: j: B
    """

    if negatives is not None:
        m = negatives.shape[1]
        # r_sim = torch.einsum('be,be->b', [reconstruction, representation]).repeat(m, 1)
        # n_sim = torch.einsum('be,bme->mb', [reconstruction, negatives])

        r_sim_ = F.cosine_similarity(reconstruction, sentence_weighted_average, dim=-1).repeat(m, 1)
        n_sim_ = F.cosine_similarity(reconstruction.unsqueeze(1).repeat(1, m, 1), negatives, dim=-1).t()

        # r_sim_ = (reconstruction.norm(dim=-1) * representation).repeat(m, 1)
        # n_sim_ = torch.einsum('b,bm->mb', [reconstruction.nomr(dim=-1), negatives.norm(dim=-1)])
    else:
        r_sim = torch.einsum('be,be->b', [reconstruction, sentence_weighted_average])
        n_sim = 0

    # j = F.relu(1 - r_sim + n_sim).sum()
    j = F.relu(1 - r_sim_ + n_sim_).sum()

    if negatives is not None:
        j = j / m

    return j


def vae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD, MSE, KLD


def reg_loss(t):
    """

    :param t: ET
    :return: 1
    """
    t_dot_tt = (t @ t.t()) / (t.norm(dim=-1, keepdim=True) @ t.norm(dim=-1, keepdim=True).t())
    t_dot_tt = t_dot_tt - torch.eye(t_dot_tt.shape[0]).to(device)
    u = t_dot_tt.norm()
    return u


model = AspectExtractor(asp_dim=20)

model = model.to(device)

c_loss = nn.CrossEntropyLoss(reduction='sum')

metrics_history_all = []

optimizer = torch.optim.Adam(model.parameters())
metrics_history = []
progress_bar = tqdm(range(1, N_EPOCHS + 1))
for i_epoch in progress_bar:
    model.train()
    loss_0_total = 0
    loss_1_total = 0
    loss_2_total = 0
    loss_3_total = 0
    accu_total = 0
    total = 0
    # progress_bar = tqdm(train_data_loader)
    for i, (x, y) in enumerate(train_data_loader):
        x = x.to(device)
        y = y.to(device)
        # for x, y in progress_bar:
        optimizer.zero_grad()
        batch_size = y.size(0)
        reconstruction, mu, logvar, z, sentence_weighted_average = model(x)

        # negatives = random_train_sample(batch_size * 20)
        # with torch.no_grad():
        #     negatives = model.aspect.weighted_avg(model.aspect.emb(negatives), negatives == PAD)[0].view(batch_size, 20,
        #                                                                                                  -1)  # BME
        #
        # loss_0 = margin_loss(reconstruction, sentence_weighted_average, negatives)
        loss_0 = torch.zeros(1).to(device)
        loss_1, mse, kld = vae_loss(reconstruction, sentence_weighted_average, mu, logvar)
        # loss_1 = torch.zeros(1).to(device)
        loss_2 = reg_loss(model.aspect.T)
        # loss_2 = torch.zeros(1).to(device)
        # loss_3 = c_loss(out, y)
        loss_3 = torch.zeros(1).to(device)

        loss = loss_0 + loss_1 + 10 * loss_2 + loss_3

        loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     accu = (out.argmax(1) == y).float().sum().item()
        accu = 0

        loss_0_total += loss_0.item()
        loss_1_total += loss_1.item()
        loss_2_total += loss_2.item()
        loss_3_total += loss_3.item()
        accu_total += accu
        total += batch_size

    model.eval()

    accu_total_test = 0
    total_test = 0
    # progress_bar = tqdm(test_data_loader)
    # for i, (x, y) in enumerate(test_data_loader):
    #     # for x, y in progress_bar:
    #
    #     batch_size = y.size(0)
    #     reconstruction, representation, attention, probs, out = model(x)
    #
    #     with torch.no_grad():
    #         accu = (out.argmax(1) == y).float().sum().item()
    #
    #     accu_total_test += accu
    #     total_test += batch_size
    #
    metrics = (
        loss_0_total / total,
        loss_1_total / total,
        loss_2_total / total,
        loss_3_total / total,
        # accu_total / total,
        # accu_total_test / total_test,
    )

    progress_bar.set_description("{:.3f} {:.3f} {:.3f} {:.3f}".format(*metrics))

    metrics_history.append(metrics)





def topic_words(n_top_words=10):
    sims = (model.aspect.T.detach() @ model.aspect.emb.weight.t().detach()) / (
            model.aspect.T.detach().norm(dim=-1, keepdim=True).detach() @ model.aspect.emb.weight.detach().norm(dim=-1,
                                                                                                                keepdim=True).t())
    sims[torch.isnan(sims)] = -1

    sims = sims.cpu()
    top_words = sims.sort(dim=-1, descending=True)[1][:,:n_top_words]
    for k, beta_k in enumerate(top_words):
        topic_words = [itos[w_id.item()] for w_id in beta_k]
        print('Topic {}: {}'.format(k, ' '.join(topic_words)))
    return top_words


top_words = topic_words(15)


# %%

inverse_index = defaultdict(list)
for word in tqdm(range(VOCAB_LEN)):
    for did, doc in enumerate(train_dataset.texts):
        if word in doc:
            inverse_index[word].append(did)

#%%


def co_document_frequency(w1, w2):
    return len(set(inverse_index[w1]).intersection(set(inverse_index[w2])))


def document_frequency(w1):
    return len(inverse_index[w1])


all_scores = []
for topic in tqdm(top_words.numpy()):
    score = 0
    for w1, w2 in combinations(topic, 2):
        score += np.log((co_document_frequency(w1, w2) + 1)/(document_frequency(w1) + document_frequency(w1) + 1))
    all_scores.append(score)
all_scores = np.array(all_scores)
print(all_scores)
print(all_scores.mean())