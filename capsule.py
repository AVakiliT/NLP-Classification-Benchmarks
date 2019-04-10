import torch
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm

# %%
torch.manual_seed(1)
BATCH_SIZE = 128
LR = 1e-3
N_EPOCHS = 30
MIN_FREQ = 8
EMB_DIM = 100
HIDDEN_DIM = 100
MAX_LEN = 800
EPSILON = 1e-13
NUM_CLASSES = 50
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

import torch
from torch import nn
from torch.nn import functional as F


class FirstLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=100, out_channels=64, kernel_size=i, padding=i // 2) for i in [1, 3, 5]])
        self.rnn = nn.GRU(100, 96, bidirectional=True)

    def forward(self, x):
        # return torch.cat([F.relu(c(x.transpose(1, 2))) for c in self.convs], 1).transpose(1, 2)
        return self.rnn(x)[0]


class PrimaryCaps(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_attend = nn.Sequential(
            nn.Linear(192, 50),
            nn.Tanh(),
            nn.Linear(50, 12),
            nn.Softmax(-1)
        )

    def att_reg(self, a):
        return (torch.einsum('btr,bts->brs', [a, a]) - torch.eye(a.shape[-1], device=device)).norm(dim=(-2, -1)).mean()

    def forward(self, xx, mask):
        att = self.self_attend(xx)  # BTH -> BTR
        m = torch.einsum('bth,btr->brh', [xx, att])
        att_reg = self.att_reg(att)
        return squash(m), att_reg


class DigitCaps(nn.Module):

    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(12, 192, 50, 16))

    def forward(self, x):
        u_hat = torch.einsum('bri,rico->brco', [x, self.W])
        b = torch.zeros(*u_hat.shape[:-1], device=device)  # brc
        for i in range(3):
            c = F.softmax(b, -1)
            s = torch.einsum('brc,brco->bco', [c, u_hat])
            v = squash(s)
            if i != 2:
                b = b + torch.einsum('brco,bco->brc', [u_hat, v])
            else:
                return v


class CapsNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.first_layer = FirstLayer()
        self.primary = PrimaryCaps()
        self.digits = DigitCaps()
        self.emb = nn.Embedding(len(TEXT.vocab), EMB_DIM, padding_idx=1)

    def forward(self, x):
        mask = data[0] == PAD
        xx = self.emb(x)
        xx, att_reg = self.primary(self.first_layer(xx), mask)
        return self.digits(xx), att_reg


def norm_squared(tensor):
    norm = tensor.pow(2).sum(-1, keepdim=True)
    return norm


def squash(tensor):
    norm = norm_squared(tensor)
    return (norm / (norm + 1)) * (tensor / norm.sqrt())


def margin_loss(v, target_one_hot):
    v_mag = norm_squared(v).sqrt()
    zero = torch.zeros(1, device=device)
    left = torch.max((0.9 - v_mag.squeeze(-1)), zero) ** 2
    right = torch.max((v_mag.squeeze(-1) - .1), zero) ** 2

    loss = target_one_hot * left + 0.5 * (1.0 - target_one_hot) * right

    loss = loss.sum()

    return loss


def one_hot(y):
    return torch.zeros(y.shape[0], 50, device=device).scatter_(-1, y.unsqueeze(-1), 1)


model = CapsNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
reconstruction_criterion = nn.MSELoss(reduction='sum')

pbar = tqdm(range(1, N_EPOCHS))
for i_epoch in pbar:
    model.train()
    loss_total = 0
    acc_total = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_data_loader):
        optimizer.zero_grad()
        output, att_reg = model(data[0])
        target_one_hot = one_hot(target)

        loss = margin_loss(output, target_one_hot) #+ 5e-4 * att_reg

        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        acc_total += (torch.argmax(norm_squared(output), 1).squeeze(-1).long() == target.long()).sum().item()
        total += target.shape[0]

    model.eval()
    loss_total_test = 0
    acc_total_test = 0
    total_test = 0
    for batch_idx, (data, target) in enumerate(test_data_loader):
        output, att_reg = model(data[0])
        target_one_hot = one_hot(target)

        loss = margin_loss(output, target_one_hot) #+ 5e-4 * att_reg

        loss_total_test += loss.item()
        acc_total_test += (torch.argmax(norm_squared(output), 1).squeeze(-1).long() == target.long()).sum().item()
        total_test += target.shape[0]

    pbar.set_description(
        '[ {:02d} / {:02d} ][ TRAIN LSS: {:.3f} ACC: {:.3f} ][ TEST LSS: {:.3f} ACC: {:.3f} ]'.format(
            i_epoch, N_EPOCHS,
            loss_total / total, acc_total / total,
            loss_total_test / total_test, acc_total_test / total_test
        ))

