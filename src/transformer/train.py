"""
Training script for the Decoder-Only Transformer.

Implements:
  - AdamW with weight decay applied only to 2D matrices (not biases/LayerNorm).
  - Linear warmup + cosine decay learning rate schedule.
  - Mixed precision via torch.cuda.amp (autocast + GradScaler for fp16).
  - Gradient clipping.
"""
import os
import sys
import math
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.transformer.model import GPT
from src.transformer.dataset import get_dataloaders
from src.utils.seed import set_seed


def configure_optimizers(model, weight_decay, learning_rate, betas=(0.9, 0.95)):
    """
    Builds AdamW parameter groups:
      - 2D params (Linear/Embedding weights) get weight_decay.
      - 1D params (biases, LayerNorm gains) get NO weight decay.
    """
    decay_params = []
    no_decay_params = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """
    Linear warmup -> cosine decay to min_lr.
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    # Cosine decay
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, val_loader, eval_iters, device, amp_dtype):
    """Estimates validation loss over eval_iters batches."""
    model.eval()
    losses = []
    it = iter(val_loader)
    for _ in range(eval_iters):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(config_path):
    # ----- Load config -----
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    set_seed(config["seed"])

    device = torch.device(
        config["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ----- Data -----
    train_loader, val_loader, vocab_size, stoi, itos = get_dataloaders(config)
    print(f"Vocab size: {vocab_size}")

    # ----- Model -----
    model = GPT(
        vocab_size=vocab_size,
        n_embd=config["model"]["n_embd"],
        n_head=config["model"]["n_head"],
        n_layer=config["model"]["n_layer"],
        block_size=config["data"]["block_size"],
        dropout=config["model"]["dropout"],
        bias=config["model"]["bias"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # ----- Optimizer -----
    optimizer = configure_optimizers(
        model,
        weight_decay=config["training"]["weight_decay"],
        learning_rate=config["training"]["learning_rate"],
    )

    # ----- AMP setup -----
    use_amp = config["training"]["use_amp"] and device.type == "cuda"
    amp_dtype = None
    scaler = None
    if use_amp:
        dtype_str = config["training"]["amp_dtype"]
        if dtype_str == "bf16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            scaler = None  # bf16 doesn't need GradScaler
        else:
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
        print(f"AMP enabled with dtype={amp_dtype}")

    # ----- Training loop -----
    os.makedirs(config["training"]["save_dir"], exist_ok=True)
    os.makedirs(config["training"]["log_dir"], exist_ok=True)

    max_steps = config["training"]["max_steps"]
    warmup_steps = config["training"]["warmup_steps"]
    max_lr = config["training"]["learning_rate"]
    min_lr = config["training"]["min_lr"]
    grad_clip = config["training"]["grad_clip"]
    eval_interval = config["training"]["eval_interval"]
    eval_iters = config["training"]["eval_iters"]

    model.train()
    step = 0
    t0 = time.time()
    pbar = tqdm(total=max_steps, desc="Training")

    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Update learning rate per step
            lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    _, loss = model(x, y)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            else:
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Logging
            if step % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
            pbar.update(1)

            # Periodic validation
            if step > 0 and step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, eval_iters, device, amp_dtype)
                print(f"\n[step {step}] train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
                # Save checkpoint
                ckpt_path = os.path.join(config["training"]["save_dir"], f"gpt_step_{step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": config,
                    "stoi": stoi,
                    "itos": itos,
                }, ckpt_path)

            step += 1
    pbar.close()

    # Final save
    final_path = os.path.join(config["training"]["save_dir"], "gpt_final.pt")
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "stoi": stoi,
        "itos": itos,
    }, final_path)
    print(f"\nTraining done in {time.time()-t0:.1f}s. Final checkpoint: {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/transformer.yaml")
    args = parser.parse_args()
    train(args.config)