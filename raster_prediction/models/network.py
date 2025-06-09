import math
import torch
import torch.nn as nn

import models.modules as modules
import models.transformers as transformers
import models.encodings as encodings


class Transformer(nn.Module):
    def __init__(self, in_channels, d_model, input_shape, nhead, dim_feedforward, num_layers, encoder, decoder,
                 transformer):
        super().__init__()

        src_seq_len = int((input_shape[0] / 4) * (input_shape[1] / 4))

        self.d_model = d_model
        self.input_shape = input_shape

        self.pos_encoder = encodings.PositionalEncoding(d_model=d_model, seq_len=src_seq_len)
        self.encoder = getattr(modules, encoder)(in_channels=in_channels, out_channels=d_model)
        self.decoder = getattr(modules, decoder)(in_channels=d_model, out_channels=in_channels)
        self.transformer = getattr(transformers, transformer)(d_model=d_model, nhead=nhead,
                                                              dim_feedforward=dim_feedforward, num_layers=num_layers,
                                                              src_seq_len=src_seq_len)

    def forward(self, x):
        x = self.encoder(x)  # (B, 64, 13, 13)
        input_shape = x.shape
        x = x.flatten(2).permute(0, 2, 1) * math.sqrt(self.d_model)  # (B, 64, 169)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).reshape(input_shape)
        x = self.decoder(x)
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, in_channels, d_model, input_shape, nhead, dim_feedforward, num_layers, encoder, decoder,
                 transformer):
        super().__init__()

        src_seq_len = int((input_shape[0] / 4) * (input_shape[1] / 4)) * 2

        self.d_model = d_model
        self.input_shape = input_shape

        self.encoder = getattr(modules, encoder)(in_channels=in_channels, out_channels=d_model)
        self.decoder = getattr(modules, decoder)(in_channels=d_model, out_channels=in_channels)
        self.transformer = getattr(transformers, transformer)(d_model=d_model, nhead=nhead,
                                                              dim_feedforward=dim_feedforward, num_layers=num_layers,
                                                              src_seq_len=src_seq_len)

    """
    def forward(self, seq):

        seq = [seq[:, i, ...] for i in range(seq.shape[1])]

        for x in seq:

            x = self.encoder(x)  # (B, 64, 13, 13)
            input_shape = x.shape
            x = x.flatten(2).permute(0, 2, 1) * math.sqrt(self.d_model)  # (B, 64, 169)
            x = self.pos_encoder(x)
            x = self.transformer(x)
            x = x.permute(0, 2, 1).reshape(input_shape)
            x = self.decoder(x)

        return x
    """

    # def forward(self, seq):
    #     b, s, c, w, h = seq.shape
    #     x = seq.reshape(b * s, c, w, h)
    #     x = self.encoder(x)  # (B, 64, 13, 13)
    #     enc_out_shape = x.shape
    #     x = x.reshape(b, s, enc_out_shape[-3], enc_out_shape[-2], enc_out_shape[-1]).reshape(b, s, enc_out_shape[-3], -1)
    #     x = x.transpose(2, 1).reshape(b, self.d_model, -1).transpose(2, 1)
    #     x = self.transformer(x)  # <-
    #     x = x.transpose(1, 2).reshape(b, self.d_model, s, enc_out_shape[-2], enc_out_shape[-1]).transpose(2, 1)
    #     x = x.reshape(b * s, self.d_model, enc_out_shape[-2], enc_out_shape[-1])
    #     x = self.decoder(x)
    #     x = x.reshape(seq.shape)
    #     return x

    def forward(self, seq):

        seq = [seq[:, i, ...] for i in range(seq.shape[1])]

        states = []
        for x in seq:
            x = self.encoder(x)
            x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[2])
            states.append(x)

        states = torch.concat(states, dim=-1).permute(0, 2, 1)
        x = self.transformer(states) # Removed
        x = x.transpose(1, 2).reshape(x.shape[0], len(seq), x.shape[2], int(x.shape[1]/len(seq)))
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], int(math.sqrt(x.shape[3])), int(math.sqrt(x.shape[3])))

        states = []
        for i in range(x.shape[1]):
            x_i = self.decoder(x[:, i, ...])
            states.append(x_i.unsqueeze(1))

        x = torch.concat(states, dim=1)
        return x


def main():

    arguments = {"in_channels": 3,  # out_channels same as in_channels
                 "d_model": 64,  # d_model is the encoding input size of the transformer
                 "input_shape": (64, 256),
                 "nhead": 8,
                 "dim_feedforward": 512,
                 "num_layers": 6,
                 "encoder": "Encoder",
                 "decoder": "Decoder",
                 "transformer": "PersonalTransformerEncoder",
                 }

    model = Transformer(**arguments)
    x = torch.zeros(8, arguments["in_channels"], arguments["input_shape"][0], arguments["input_shape"][1])
    yp = model(x)
    print(yp.shape)


if __name__ == '__main__':
    main()
