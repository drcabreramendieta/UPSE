
from torch import nn
from .positional_encoding import PositionalEncoding


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim: int = 2, d_model: int = 256, nhead: int = 8,
                 layers: int = 4, ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=ff, dropout=dropout,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pe = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x):  # x: (B, 1024, 2)
        h = self.proj(x)
        h = self.pe(h)
        h = self.encoder(h)   # no padding: todas las secuencias tienen T=1024
        h = self.norm(h)
        return h.mean(dim=1)  # mean pooling
