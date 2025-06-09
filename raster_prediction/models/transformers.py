import math
import torch
import torch.nn as nn
from models.vit_model import TransformerEncoder


class OfficialTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, src_seq_len):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)


class PersonalTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, src_seq_len):
        super().__init__()

        self.transformer = TransformerEncoder(src_seq_len=src_seq_len, d_model=d_model, N=num_layers, h=nhead, d_ff=dim_feedforward)

    def forward(self, x):
        return self.transformer(x)
