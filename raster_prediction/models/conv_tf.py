import math
import torch
import torch.nn as nn

import models.modules as modules


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int):
        super().__init__()

        self.dropout = nn.Dropout(p=0.1)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe.permute(1, 0, 2)
        return self.dropout(x)


class SpatialTransformerEncoder(nn.Module):
    def __init__(self, in_channels, d_model, input_shape, nhead, dim_feedforward, num_layers, encoder, decoder):
        super().__init__()

        self.src_seq_len = int((input_shape[0]/4) * (input_shape[1]/4))
        self.d_model = d_model
        self.input_shape = input_shape

        self.pos_encoder = PositionalEncoding(d_model=d_model, seq_len=self.src_seq_len)
        self.encoder = getattr(modules, encoder)(in_channels=in_channels, out_channels=d_model)
        self.decoder = getattr(modules, decoder)(in_channels=d_model, out_channels=in_channels)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.encoder(x)  # (B, 64, 13, 13)
        input_shape = x.shape
        x = x.flatten(2).permute(0, 2, 1) * math.sqrt(self.d_model)  # (B, 64, 169)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).reshape(input_shape)
        x = self.decoder(x)
        return x


if __name__ == '__main__':

    arguments = {"in_channels": 3,  # out_channels same as in_channels
                 "d_model": 64,  # d_model is the encoding input size of the transformer
                 "input_shape": (64, 256),
                 "nhead": 8,
                 "dim_feedforward": 512,
                 "num_layers": 6,
                 "encoder": "Encoder",
                 "decoder": "Decoder"
    }

    model = SpatialTransformerEncoder(**arguments)
    x = torch.zeros(8, arguments["in_channels"], arguments["input_shape"][0], arguments["input_shape"][1])
    yp = model(x)
    print(yp.shape)
