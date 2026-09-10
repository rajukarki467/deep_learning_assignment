"""
Dataset loader for the Decoder-Only Transformer.
Supports char-level Shakespeare (default) and a synthetic code dataset.
"""
import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare(data_dir):
    """Downloads the tiny Shakespeare text file if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, "input.txt")
    if not os.path.exists(filepath):
        print(f"Downloading Shakespeare dataset to {filepath} ...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, filepath)
    return filepath


class CharDataset(Dataset):
    """
    Character-level dataset. Each sample is a (input_seq, target_seq) pair
    where target_seq is input_seq shifted by one token (next-token prediction).
    """
    def __init__(self, text, block_size, stoi, itos):
        self.block_size = block_size
        self.stoi = stoi
        self.itos = itos
        # Encode the entire corpus as a 1D tensor of token IDs
        self.data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        # One sample per possible starting position
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x is a chunk of block_size tokens, y is the same chunk shifted by 1
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def build_char_dataset(text, block_size, train_split=0.9):
    """
    Builds train/val CharDataset objects along with vocab metadata.
    Returns: (train_ds, val_ds, vocab_size, stoi, itos)
    """
    # Build character vocabulary (sorted unique chars)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Split text into train and validation
    n = int(train_split * len(text))
    train_text, val_text = text[:n], text[n:]

    train_ds = CharDataset(train_text, block_size, stoi, itos)
    val_ds = CharDataset(val_text, block_size, stoi, itos)
    return train_ds, val_ds, vocab_size, stoi, itos


def get_dataloaders(config):
    """Builds the training and validation dataloaders from config."""
    cfg_data = config["data"]
    if cfg_data["dataset"] == "shakespeare":
        path = download_shakespeare(cfg_data["data_dir"])
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif cfg_data["dataset"] == "synthetic":
        # Synthetic code-like corpus for debugging without internet
        text = ("def foo(x):\n    return x + 1\n" * 5000)
    else:
        raise ValueError(f"Unknown dataset {cfg_data['dataset']}")

    train_ds, val_ds, vocab_size, stoi, itos = build_char_dataset(
        text, cfg_data["block_size"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_data["batch_size"],
        shuffle=True,
        num_workers=cfg_data["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_data["batch_size"],
        shuffle=False,
        num_workers=cfg_data["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader, vocab_size, stoi, itos