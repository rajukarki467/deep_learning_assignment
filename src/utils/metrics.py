"""Small metric helpers for both VAE and Transformer experiments."""
import torch
import torch.nn.functional as F


def perplexity(loss):
    """Perplexity = exp(cross-entropy loss)."""
    return torch.exp(torch.tensor(loss)).item()


def accuracy_from_logits(logits, targets):
    """Token-level accuracy for the language model."""
    preds = logits.argmax(dim=-1)
    mask = targets != -1
    correct = (preds == targets) & mask
    return correct.sum().item() / mask.sum().item()