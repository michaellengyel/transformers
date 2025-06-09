import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int):
        super().__init__()

        # self.dropout = nn.Dropout(p=0.1)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.pe = self.pe.permute(1, 2, 0).view(1, d_model, 64, 64)

    def forward(self, x):
        x = x + self.pe
        return x  # self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10 ** -6):
        super().__init__()
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        return self.norm(x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)).transpose(2, 1).reshape(x.shape)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.conv_1 = nn.Conv2d(d_model, d_ff, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv2d(d_ff, d_model, kernel_size=1, stride=1)

    def forward(self, x):
        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_ff) -> (Batch, Seq_len, d_model)
        return self.conv_2(self.dropout(torch.relu(self.conv_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h!"
        self.d_k = d_model // h

        self.w_q = nn.Conv2d(d_model, d_model, kernel_size=4, stride=4)  # Wq
        self.w_k = nn.Conv2d(d_model, d_model, kernel_size=4, stride=4)  # Wk
        self.w_v = nn.Conv2d(d_model, d_model, kernel_size=4, stride=4)  # Wv

        self.w_o = nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=4)  # Wv

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.reshape(query.shape[0], query.shape[1], -1).view(query.shape[0], self.h, self.d_k, query.shape[-1]*query.shape[-2]).transpose(-2, -1)
        key = key.reshape(key.shape[0], key.shape[1], -1).view(key.shape[0], self.h, self.d_k, key.shape[-1]*key.shape[-2]).transpose(-2, -1)
        value = value.reshape(value.shape[0], value.shape[1], -1).view(value.shape[0], self.h, self.d_k, value.shape[-1]*value.shape[-2]).transpose(-2, -1)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_len, d_k) -> (Batch, Seq_len, h, d_k) -> (Batch, Seq_len, d_model)
        x = x.transpose(-2, -1).contiguous().view(x.shape[0], self.h * self.d_k, -1).view(x.shape[0], self.h * self.d_k, 16, 16)

        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerEncoder(nn.Module):
    def __init__(self, src_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
        super().__init__()

        # Create encoder blocks
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)
        self.encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

        # Embedder and deembedder
        self.embedder = nn.Conv2d(3, d_model, kernel_size=1, stride=1)  # Wv
        self.unembedder = nn.Conv2d(d_model, 3, kernel_size=1, stride=1)  # Wv

        # Create the positional encoding layer
        self.src_pos = PositionalEncoding(d_model, 64*64)

    def forward(self, src):
        src = self.embedder(src)
        src = self.src_pos(src)
        src = self.encoder(src, None)
        src = self.unembedder(src)
        src = torch.sigmoid(src)
        return src


if __name__ == '__main__':

    model = TransformerEncoder(src_seq_len=169, d_model=64, N=8, h=8, dropout=0.1, d_ff=256)
    print(f'Number of parameters: {torch.nn.utils.parameters_to_vector(model.parameters()).numel()}')

    x = torch.zeros(8, 3, 64, 64)

    yp = model(x)
    print(yp)
