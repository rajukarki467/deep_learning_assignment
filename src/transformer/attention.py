"""
Manual Causal Multi-Head Self-Attention.

Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + M ) V
where M is a lower-triangular mask with 0 on/below the diagonal and -inf above.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    A single causal self-attention layer.

    Instead of separate Q/K/V projections, we use one Linear that produces
    3 * n_embd outputs, then split into Q, K, V. This is more efficient and
    matches the GPT-2 implementation.
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1, bias=False):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # Single projection for Q, K, V (three times the embedding dim)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Pre-register the causal mask as a buffer (not a parameter).
        # Shape (1, 1, T, T), lower triangular of ones -> we invert for masking.
        # register_buffer moves it with .to(device) and saves in state_dict.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(self, x):
        """
        x: (B, T, C) tensor of token embeddings.
        Returns: (B, T, C) tensor of attention outputs.
        """
        B, T, C = x.shape

        # 1) Compute Q, K, V for the entire sequence in one shot.
        #    qkv: (B, T, 3*C)
        qkv = self.c_attn(x)
        # Split into 3 chunks along the last dim, each (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 2) Reshape into (B, n_head, T, head_dim) so each head attends independently.
        #    (B, T, n_head, head_dim) -> transpose to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3) Scaled dot-product attention.
        #    att: (B, n_head, T, T)
        #    Q K^T / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 4) Apply causal mask: positions where mask==0 (upper triangular) are set to -inf.
        #    bias is (1, 1, T, T) with 1s on/below diagonal and 0s above.
        #    masked_fill where bias == 0 with -inf.
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 5) Softmax over the last dim (keys).
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6) Weighted sum of values: (B, n_head, T, head_dim)
        y = att @ v

        # 7) Concatenate heads back: transpose -> (B, T, n_head, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8) Output projection + dropout
        y = self.resid_dropout(self.c_proj(y))
        return y