import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.data import Field, TabularDataset, BucketIterator, Example, NestedField, Dataset
from torchtext.vocab import GloVe
from tqdm import tqdm
import pandas as pd

# %%

BATCH_SIZE = 256
LR = 1e-3
N_EPOCHS = 100
MIN_FREQ = 8
EMB_DIM = 100
HIDDEN_DIM = 100
MAX_LEN = 35
# folder = '/home/amir/IIS/Datasets/new_V2/Ubuntu_corpus_V2/'
# device = torch.device('cpu')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TEXT = Field(sequential=True, use_vocab=True, fix_length=MAX_LEN, tokenize=lambda x: x.split(),
             # include_lengths=True,
             batch_first=True, pad_first=True, truncate_first=True)

NESTED_TEXT = NestedField(TEXT,
                          fix_length=MAX_LEN,
                          truncate_first=True,
                          pad_first=True)

LABEL = Field(sequential=False, use_vocab=False, batch_first=True)

columns = [('text', TEXT),
           ('label', LABEL),
           ('text_split', NESTED_TEXT)
           ]


train_df = pd.read_csv('train_clean_split.csv', converters={'text_split': lambda x: x.split('\t')})
train = Dataset([Example.fromlist(row, columns) for _, row in train_df.iterrows()], columns)

test_df = pd.read_csv('test_clean_split.csv', converters={'text_split': lambda x: x.split('\t')})
test = Dataset([Example.fromlist(row, columns) for _, row in test_df.iterrows()], columns)


# train = TabularDataset(
#     path='train_clean.csv',
#     format='csv',
#     fields=columns,
#     skip_header=True
# )
#
# test = TabularDataset(
#     path='test_clean.csv',
#     format='csv',
#     fields=columns,
#     skip_header=True
# )

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
            yield batch.text, batch.text_split, batch.label

    def __len__(self):
        return len(self.iterator)


train_data_loader = TrainIterWrap(train_iter)
test_data_loader = TrainIterWrap(test_iter)


# %%
class TextCNN(nn.Module):

    def __init__(self):
        super().__init__()
        N_FILTERS = 50
        SIZES = [1, 2, 3, 5]
        self.emb = nn.Embedding(len(TEXT.vocab), EMB_DIM, padding_idx=1)
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, N_FILTERS, (i, EMB_DIM)),
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


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(TEXT.vocab), EMB_DIM, padding_idx=1)
        FILTER_SIZE = 3
        POOLING_SIZE = 3
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=EMB_DIM, out_channels=EMB_DIM, kernel_size=FILTER_SIZE),
                nn.ReLU(),
                nn.MaxPool1d(POOLING_SIZE)
            ) for _ in range(3)])
        self.final = nn.Linear(EMB_DIM, 50)

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
        self.emb = nn.Embedding(len(TEXT.vocab), EMB_DIM, padding_idx=1)
        self.RNN = nn.GRU(EMB_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.final = nn.Linear(HIDDEN_DIM, 50, bias=False)

    def forward(self, x, x_len, mask):
        x = self.emb(x)

        x = x.mean(-1)
        x = self.final(x).squeeze()
        return x


class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(TEXT.vocab), EMB_DIM, padding_idx=1)
        self.RNN = nn.GRU(EMB_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.RNN2 = nn.GRU(HIDDEN_DIM * 2, HIDDEN_DIM * 2, batch_first=True, bidirectional=True)
        self.final = nn.Linear(HIDDEN_DIM * 4, 50, bias=False)

    def sentence(self, s):
        s = self.emb(s)
        s = self.RNN(s)[0]
        s = s.mean(-2)
        return s

    def forward(self, _, _x_split, _mask):

        _x_split = _x_split.transpose(0, 1)
        xs = torch.stack([self.sentence(s) for s in _x_split]).transpose(0, 1)
        xs = self.RNN2(xs)[0]

        xs = xs.mean(-2)
        xs = self.final(xs)
        return xs


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
        s = s / (s.sum(dim=-1, keepdim=True) + 1e-13)
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
        self.emb = nn.Embedding(len(TEXT.vocab), EMB_DIM, padding_idx=1)
        self.RNN = nn.GRU(EMB_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
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


# models = [
#     AvgEmbClassifier().to(device),
#     LSTMClassifier().to(device),
#     AttentionClassifier(False).to(device),
#     AttentionClassifier(True).to(device)
# ]


models = [
    LSTMClassifier().to(device),
    # AttentionClassifier(False, False).to(device),
    # AttentionClassifier(False, True).to(device),
    # AttentionClassifier(True, False).to(device),
    # AttentionClassifier(True, True).to(device),
    # CNN().to(device),
    # TextCNN().to(device),
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
        for x, x_split, y in train_data_loader:
            optimizer.zero_grad()
            mask = x != PAD
            prediction = model(x, x_split, mask)
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
        for x, x_split, y in test_data_loader:
            mask = x != PAD
            prediction = model(x, x_split, mask)
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
# plt.show()
plt.savefig('acc3.png', dpi=300, bbox_inches='tight')
