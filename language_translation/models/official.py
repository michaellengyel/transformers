import math
import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # Tensor is not trained/changed
        return self.dropout(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, n=6, h=8, dropout=0.1, d_ff=2048):
        super().__init__()

        # Create encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=h, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n)

        # Create decoder blocks
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=h, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n)

        self.src_embed = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        self.src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        nn.init.xavier_uniform_(self.src_embed.embedding.weight)
        nn.init.xavier_uniform_(self.tgt_embed.embedding.weight)


    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        src_mask = ~src_mask.bool()[:, 0, 0, :]
        return self.encoder(src=src, src_key_padding_mask=src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt_mask = ~tgt_mask.bool()[:, 0, 0, :]
        src_mask = ~src_mask.bool()[:, 0, 0, :]
        mask = torch.triu(torch.ones((tgt.size(1), tgt.size(1)), device=tgt.device), diagonal=1).bool()
        output = self.decoder(tgt=tgt, memory=encoder_output, tgt_mask=mask, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
        return output

    def project(self, x):
        return self.projection_layer(x)

    def greedy_decode(self, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
        sos_index = tokenizer_tgt.token_to_id('[SOS]')
        eos_index = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it got every token we get from the decoder
        encoder_output = self.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_index).type_as(source).to(device)

        while True:
            if decoder_input.size(1) == max_len:
                break

            tgt_mask = torch.triu(torch.ones((decoder_input.size(1), decoder_input.size(1)), device=device), diagonal=1).bool()

            # Calculate the output
            tgt = self.tgt_embed(decoder_input)
            tgt = self.tgt_pos(tgt)
            out = self.decoder(tgt=tgt, memory=encoder_output, tgt_mask=tgt_mask)

            # Get the next token
            logits = self.project(out[:, -1])
            _, next_word = torch.max(logits, dim=1)
            decoder_input = torch.cat([decoder_input, next_word.unsqueeze(1)], dim=1)  # Append along sequence dimension

            if next_word == eos_index:
                break

        return decoder_input.squeeze(0)
