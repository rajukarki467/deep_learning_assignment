import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.vae.model import VAE
from src.vae.dataset import get_dataloader
from src.utils.visualization import plot_reconstruction

def evaluate(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = VAE(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim']
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Get one batch of data
    dataloader = get_dataloader(config)
    data, _ = next(iter(dataloader))
    data = data.to(device)
    
    with torch.no_grad():
        recon, _, _ = model(data)
        
    # Save figure
    os.makedirs("./results/figures", exist_ok=True)
    plot_reconstruction(data, recon, "./results/figures/reconstruction_comparison.png")
    print("Reconstruction plot saved to results/figures/reconstruction_comparison.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/vae.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)