from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, EMBEDDING_DIM=None, HIDDEN_DIM=None, NUM_CLASSES=None):
        super().__init__()
        self.emb = get_emb()
        self.RNN = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True, num_layers=2)
        self.final = nn.Linear(HIDDEN_DIM * 2, NUM_CLASSES, bias=False)

    def forward(self, x, x_len, mask):
        x = self.emb(x)

        # x = pack_padded_sequence(x, x_len, batch_first=True)
        x = self.RNN(x)[0]
        # x = pad_packed_sequence(x, batch_first=True)[0]

        x = x[:, -1, :]
        x = self.final(x).squeeze()
        return x
