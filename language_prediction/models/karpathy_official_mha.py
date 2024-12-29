import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttentionWrapper(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, seq_len, d_model, h, dropout):
        super().__init__()
        self.register_buffer('triu', torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())
        self.sa = nn.MultiheadAttention(embed_dim=d_model, num_heads=h, dropout=dropout, batch_first=True)

    def forward(self, x):
        b, s, d_model = x.shape
        mask = self.triu[:s, :s]
        x = self.sa(x, x, x, attn_mask=mask, need_weights=False)[0]
        return x


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, d_model, dropout, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, seq_len, d_model, h, dropout, d_ff):
        # d_model: embedding dimension, h: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttentionWrapper(seq_len, d_model, h, dropout)
        self.ffwd = FeedFoward(d_model, dropout, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.sa(self.ln1(x)))
        x = x + self.dropout2(self.ffwd(self.ln2(x)))
        return x

class GPT(nn.Module):

    def __init__(self, vocab_size, seq_len, d_model, n, h, dropout, d_ff):
        super().__init__()
        self.seq_len = seq_len
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(seq_len, d_model)
        self.blocks = nn.Sequential(*[Block(seq_len=seq_len, d_model=d_model, h=h, dropout=dropout, d_ff=d_ff) for _ in range(n)])
        self.ln_f = nn.LayerNorm(d_model)  # final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, s = idx.shape
        device = idx.device

        # idx and targets are both (b, s) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (b, s, d_model)
        pos_emb = self.position_embedding_table(torch.arange(s, device=device))  # (s, d_model)
        x = tok_emb + pos_emb  # (b, s, d_model)
        x = self.blocks(x)  # (b, s, d_model)
        x = self.ln_f(x)  # (b, s, d_model)
        logits = self.lm_head(x)  # (b, s, vocab_size)

        if targets is None:
            loss = None
        else:
            b, s, d_model = logits.shape
            logits = logits.view(b * s, d_model)
            targets = targets.view(b * s)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (b, s) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.seq_len:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (b, d_model)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (b, d_model)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (b, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (b, s+1)
        return idx
