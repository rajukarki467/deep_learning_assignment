"""
Decoder-Only Transformer (GPT-style) built from PyTorch primitives.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer.attention import CausalSelfAttention


class MLP(nn.Module):
    """Feed-forward network used inside each Transformer block (GELU activation)."""
    def __init__(self, n_embd, dropout=0.1, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block:
        x = x + Attn(LN(x))
        x = x + MLP(LN(x))
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1, bias=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x):
        # Pre-LN residual around attention
        x = x + self.attn(self.ln_1(x))
        # Pre-LN residual around MLP
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    Miniature GPT-style decoder-only Transformer.
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1, bias=False):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        # Token embedding + Learned positional embedding
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout, bias)
            for _ in range(n_layer)
        ])

        # Final LayerNorm + LM head (no bias, weight-tied with wte optionally)
        self.ln_f = nn.LayerNorm(n_embd, bias=bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying between token embedding and LM head (GPT-2 trick)
        self.lm_head.weight = self.wte.weight

        # Initialize weights (GPT-2 style: normal with std 0.02, scaled residual proj)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layer) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token ids
        targets: (B, T) next-token targets, or None for inference
        Returns: (logits, loss) where loss is None if targets is None.
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

        # Token + positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)                 # (B, T, C)
        pos_emb = self.wpe(pos)                 # (T, C) -> broadcasts to (B, T, C)
        x = self.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)                        # (B, T, C)
        logits = self.lm_head(x)                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive sampling. Supports greedy, temperature, and top-k.
        idx: (B, T) context token ids.
        Returns: (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Take logits at the last position
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k filtering: keep only the k highest logits
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set everything below the k-th largest to -inf
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx