
import torch
from torch import nn


class LSTMBackbone(nn.Module):
    def __init__(self, input_dim: int = 2, hidden: int = 256, layers: int = 2,
                 bidir: bool = True, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden, num_layers=layers,
                           batch_first=True, bidirectional=bidir, dropout=dropout)
        self.out_dim = hidden * (2 if bidir else 1)

    def forward(self, x):  # x: (B, 1024, 2)
        _, (h, _) = self.rnn(x)
        if self.rnn.bidirectional:
            return torch.cat([h[-2], h[-1]], dim=-1)
        return h[-1]
