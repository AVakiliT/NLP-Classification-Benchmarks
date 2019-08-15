class ConvolutionEncoder(nn.Module):

    def __init__(self, args):
        super(ConvolutionEncoder, self).__init__()

        self.conv1 = nn.Conv1d(100, 64, (args.filter_size, args.word_dim), stride=args.stride)
        self.conv2 = nn.Conv1d(args.feature_maps[0], args.feature_maps[1], (args.filter_size, 1), stride=args.stride)
        self.conv3 = nn.Conv1d(args.feature_maps[1], args.feature_maps[2], (args.filter_size, 1), stride=args.stride)

        self.relu = nn.ReLU()

    def forward(self, x):
        # reshape for convolution layer
        x.unsqueeze_(1)

        h1 = self.relu(self.conv1(x))
        h2 = self.relu(self.conv2(h1))
        h3 = self.relu(self.conv3(h2))

        # (batch, feature_maps[2])
        h3.squeeze_()
        if len(h3.size()) < 2:
            h3.unsqueeze_(0)
        return h3
