import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe
from tqdm import tqdm

# %%

BATCH_SIZE = 64
LR = 1e-3
N_EPOCHS = 20
MIN_FREQ = 8
EMBEDDING_DIM = 100
HIDDEN_DIM = 100
MAX_LEN = 800
EPSILON = 1e-13
# folder = '/home/amir/IIS/Datasets/new_V2/Ubuntu_corpus_V2/'
# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TEXT = Field(sequential=True, use_vocab=True, fix_length=MAX_LEN, tokenize=lambda x: x.split(), include_lengths=True,
             batch_first=True, pad_first=True, truncate_first=True)

LABEL = Field(sequential=False, use_vocab=False, batch_first=True)

columns = [('text', TEXT),
           ('label', LABEL)]

train = TabularDataset(
    path='train_clean.csv',
    format='csv',
    fields=columns,
    skip_header=True
)

test = TabularDataset(
    path='test_clean.csv',
    format='csv',
    fields=columns,
    skip_header=True
)

TEXT.build_vocab(train, min_freq=MIN_FREQ,
                 # vectors=GloVe(name='6B', dim=200, cache='/home/amir/IIS/Datasets/embeddings')
                 )

PAD = 1
# %%
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

#%%
# w2v = gensim.models.Word2Vec.load('msdialog_' + str(EMBEDDING_DIM) + '.w2v')
#
# embedding_weights = torch.zeros(len(TEXT.vocab), EMBEDDING_DIM)
# init.normal_(embedding_weights)
#
# for i, word in enumerate(TEXT.vocab.itos):
#     if word in w2v.wv and i != PAD:
#         embedding_weights[i] = torch.Tensor(w2v.wv[word])
#
# embedding_weights.to(device)
def get_emb():
    return nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=PAD)
# %%
class TextCNN(nn.Module):

    def __init__(self):
        super().__init__()
        N_FILTERS = 50
        SIZES = [1, 2, 3, 5]
        self.emb = nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=1)
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, N_FILTERS, (i, EMBEDDING_DIM)),
                nn.ReLU(),
                nn.MaxPool2d((MAX_LEN - i + 1, 1))
            )
            for i in SIZES
        ])
        self.final = nn.Linear(N_FILTERS * len(SIZES), 50)

    def forward(self, x, _, __):
        x = self.emb(x)
        x = x.unsqueeze(1)
        xs = [l(x).squeeze() for l in self.cnn]
        x = torch.cat(xs, 1)
        return self.final(x).squeeze()


class TextCnnWithFusion(nn.Module):

    def __init__(self):
        super().__init__()
        N_FILTERS = 50
        SIZES = [1, 2, 3, 5]
        self.emb = nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=1)
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, N_FILTERS, (i, EMBEDDING_DIM)),
                nn.ReLU(),
                nn.MaxPool2d((MAX_LEN - i + 1, 1))
            )
            for i in SIZES
        ])
        self.final = nn.Linear(N_FILTERS * len(SIZES), 50)

        self.cnn2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, N_FILTERS, (i, EMBEDDING_DIM)),
                nn.ReLU(),
                nn.MaxPool2d((MAX_LEN - i + 1, 1))
            )
            for i in SIZES
        ])
        self.final2 = nn.Linear(N_FILTERS * len(SIZES), 50)

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


def get_final(a, b, c):
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
        self.emb = nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=1)
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
            get_final(EMBEDDING_DIM, 100, 50),
            get_final(EMBEDDING_DIM, 100, 50),
            get_final(HIDDEN_DIM, 100, 50),
            get_final(HIDDEN_DIM, 100, 50),
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

        x_cnn = torch.cat(x_cnns, -2).transpose(1,2)

        x_rnn = self.rnn(x_emb)[0]

        representations = [x_emb, x_hat, x_cnn, x_rnn]

        representations = [i.max(-2)[0] for i in representations]

        representations = [l(i) for l, i in zip(self.finals, representations)]

        return sum(representations)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=1)
        FILTER_SIZE = 3
        POOLING_SIZE = 3
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=EMBEDDING_DIM, out_channels=EMBEDDING_DIM, kernel_size=FILTER_SIZE),
                nn.ReLU(),
                nn.MaxPool1d(POOLING_SIZE)
            ) for _ in range(3)])
        self.final = nn.Linear(EMBEDDING_DIM, 50)

    def forward(self, x, x_len, mask):
        x = self.emb(x)
        x = x.permute(0, 2, 1)
        for l in self.cnn:
            x = l(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze()
        x = self.final(x)
        return x


class AvgEmbClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=1)
        self.RNN = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.final = nn.Linear(HIDDEN_DIM, 50, bias=False)

    def forward(self, x, x_len, mask):
        x = self.emb(x)

        x = x.mean(-1)
        x = self.final(x).squeeze()
        return x


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=1)
        self.RNN = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=2)
        self.final = nn.Linear(HIDDEN_DIM * 2, 50, bias=False)

    def forward(self, x, x_len, mask):
        x = self.emb(x)

        # x = pack_padded_sequence(x, x_len, batch_first=True)
        x = self.RNN(x)[0]
        # x = pad_packed_sequence(x, batch_first=True)[0]

        x = x[:, -1, :]
        x = self.final(x).squeeze()
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
        self.emb = nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=1)
        self.RNN = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.RNN2 = nn.GRU(HIDDEN_DIM_2, HIDDEN_DIM_2, batch_first=True, bidirectional=True)
        self.W = nn.Linear(HIDDEN_DIM_2 * 2, HIDDEN_DIM_2 * 2)
        self.context_vector = Parameter(torch.randn(HIDDEN_DIM_2 * 2), requires_grad=True)
        self.final = nn.Linear(HIDDEN_DIM_2 * 2, 50)

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
class BiLSTM(nn.Module):

    def __init__(self):
        super().__init__()
        HIDDEN_DIM = 200
        self.emb = get_emb()
        self.RNN = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=2)
        self.final = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 200),
            nn.ReLU(),
            nn.Linear(200, 50)
        )

    def forward(self, x, x_len, mask):
        x = self.emb(x)
        x = self.RNN(x)[0]
        x = x.max(-2)[0]
        x = self.final(x)
        return x.squeeze()


class BiLSTMwithFusion(nn.Module):

    def __init__(self):
        super().__init__()
        HIDDEN_DIM = 200
        self.emb = nn.Embedding(len(TEXT.vocab), EMBEDDING_DIM, padding_idx=PAD)
        self.RNN = nn.GRU(2 * EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=1)
        self.RNN2 = nn.GRU(4 * HIDDEN_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=1)
        # self.final = nn.Linear(HIDDEN_DIM * 4, HIDDEN_DIM)
        self.final = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 200),
            nn.ReLU(),
            nn.Linear(200, 50)
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

# models = [
#     AvgEmbClassifier().to(device),
#     LSTMClassifier().to(device),
#     AttentionClassifier(False).to(device),
#     AttentionClassifier(True).to(device)
# ]

#%%
models = [
    # LSTMClassifier().to(device),
    # AttentionClassifier(False, False).to(device),
    # AttentionClassifier(False, True).to(device),
    # AttentionClassifier(True, False).to(device),
    # AttentionClassifier(True, True).to(device),
    # CNN().to(device),
    # TextCnnWithFusionAndContext().to(device),
    TextCNN().to(device),
    # TextCnnWithFusion().to(device),
    BiLSTM().to(device),
    # BiLSTMwithFusion().to(device)
]
criterion = nn.CrossEntropyLoss()

metrics_history_all = []
for model in models:
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

    metrics_history_all.append(metrics_history)

# %%
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

metrics_history_all_t = np.array(metrics_history_all).transpose((0, 2, 1))

plt.clf()
ax = plt.gca()
for history in metrics_history_all_t:
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(history[1], color=color)
    plt.plot(history[3], '--', color=color)

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

legends = []
for i in map(str, range(len(models))):
    for j in ['train', 'test']:
        legends.append(' '.join((i, j)))

plt.legend(legends, loc='upper left',
           bbox_to_anchor=(0, -0.2),
           fancybox=True, shadow=True, ncol=2)
plt.title("Comparison of ANN Models")
plt.show()
# plt.savefig('acc3.png', dpi=300, bbox_inches='tight')
