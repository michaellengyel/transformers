import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class EinsumAttention(nn.Module):
    """Optimized self-attention using einsum operations"""

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Create causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = head_size ** -0.5

    def forward(self, x):
        B, T, C = x.shape

        # Project queries, keys, values
        q = self.query(x)  # (B, T, hs)
        k = self.key(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # Attention scores with scaled dot product
        attn = torch.einsum('bqd,bkd->bqk', q, k) * self.scale

        # Apply causal mask
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Compute output
        out = torch.einsum('bqk,bkd->bqd', attn, v)
        return out


class MultiHeadEinsumAttention(nn.Module):
    """Multi-head attention using parallel einsum computations"""

    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        assert n_embd % num_heads == 0

        self.head_size = n_embd // num_heads
        self.num_heads = num_heads

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

        # Create causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        )

        self.scale = self.head_size ** -0.5

    def forward(self, x):
        B, T, C = x.shape

        # Project q,k,v all at once
        qkv = self.qkv(x)

        # Split into heads and separate q,k,v
        qkv = einops.rearrange(
            qkv,
            'b t (h d three) -> three b h t d',
            three=3,
            h=self.num_heads
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale

        # Apply causal mask
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Compute output
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)

        # Merge heads
        out = einops.rearrange(out, 'b h t d -> b t (h d)')

        # Final projection
        out = self.proj(out)
        out = self.dropout(out)
        return out


class EinsumMLP(nn.Module):
    """Optimized MLP using einsum"""

    def __init__(self, n_embd, dropout, expansion_factor=4):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, expansion_factor * n_embd)
        self.fc2 = nn.Linear(expansion_factor * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EinsumTransformerBlock(nn.Module):
    """Transformer block with optimized attention and MLP"""

    def __init__(self, d_model, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadEinsumAttention(d_model, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = EinsumMLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT model with optimized einsum operations"""

    def __init__(self, vocab_size, seq_len, d_model, n, h, dropout, d_ff):
        super().__init__()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EinsumTransformerBlock(d_model, h, seq_len, dropout)
            for _ in range(n)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

        # Save hyperparameters
        self.seq_len = seq_len

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        # Token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(pos)

        # Combine embeddings
        x = tok_emb + pos_emb
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate tokens using the language model"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx[:, -self.seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append new token
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
