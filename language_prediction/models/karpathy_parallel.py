import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, seq_len, d_model, h, dropout):
        super().__init__()

        self.h = h
        self.d_k = d_model // h
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(self.d_k * h, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.dropout(self.proj(out))
        return out

    def attention(self, x):
        # (b, s, d_model)
        b, s, d_model = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # (b, s, d_model) -> (b, s, h, d_k) -> (b, h, s, d_k)
        q = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)

        # Compute attention scores ("affinities")
        d_k = q.shape[-1]
        wei = q @ k.transpose(-2,-1) * d_k ** -0.5  # (b, s, d_k) @ (b, d_k, s) -> (b, s, s)
        wei = wei.masked_fill(self.tril[:s, :s] == 0, float('-inf'))  # (b, s, s)
        wei = F.softmax(wei, dim=-1)  # (b, s, s)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ v  # (b, s, s) @ (b, s, d_k) -> (b, s, d_k)

        # Reshape to (b, s, d_model)
        out = out.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, d_model, dropout, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, seq_len, d_model, h, dropout, d_ff):
        # d_model: embedding dimension, h: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(seq_len, d_model, h, dropout)
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
