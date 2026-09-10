"""
Autoregressive text generation from a trained GPT checkpoint.
Supports greedy decoding, temperature scaling, and top-k sampling.
"""
import os
import sys
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.transformer.model import GPT


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    model = GPT(
        vocab_size=len(stoi),
        n_embd=config["model"]["n_embd"],
        n_head=config["model"]["n_head"],
        n_layer=config["model"]["n_layer"],
        block_size=config["data"]["block_size"],
        dropout=0.0,  # disable dropout at inference
        bias=config["model"]["bias"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, stoi, itos, config


def encode(text, stoi):
    """Encode a string into a 1D tensor of token ids (unknown chars skipped)."""
    return torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)


def decode(ids, itos):
    """Decode a 1D tensor of token ids back to a string."""
    return "".join(itos[int(i)] for i in ids)


@torch.no_grad()
def generate_text(model, stoi, itos, prompt, max_new_tokens, temperature, top_k, device):
    """
    Runs autoregressive sampling. `temperature=0` selects greedy decoding.
    """
    if len(prompt) == 0:
        # Use a single arbitrary starting token if no prompt
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        idx = encode(prompt, stoi).unsqueeze(0).to(device)

    if temperature <= 0:
        # Greedy decoding
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.block_size:]
            logits, _ = model(idx_cond)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
    else:
        idx = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)

    return decode(idx[0].cpu(), itos)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="ROMEO: ")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stoi, itos, config = load_model(args.checkpoint, device)

    print("=" * 60)
    print(f"Prompt: {args.prompt!r}  (T={args.temperature}, top_k={args.top_k})")
    print("=" * 60)
    out = generate_text(
        model, stoi, itos, args.prompt,
        args.max_new_tokens, args.temperature, args.top_k, device,
    )
    print(out)

    print("\n" + "=" * 60)
    print("Greedy decoding demo:")
    print("=" * 60)
    out_greedy = generate_text(model, stoi, itos, args.prompt, 200, temperature=0.0, top_k=None, device=device)
    print(out_greedy)