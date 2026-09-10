import os
import sys
import yaml
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.vae.model import VAE
from src.vae.dataset import get_dataloader
from src.utils.visualization import save_image_grid, plot_interpolation

def generate_and_interpolate(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = config['model']['latent_dim']
    
    # Load Model
    model = VAE(
        in_channels=config['model']['in_channels'],
        latent_dim=latent_dim
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    os.makedirs("./results/figures", exist_ok=True)
    
    # 1. Synthetic Sampling (from Prior)
    with torch.no_grad():
        z = torch.randn(64, latent_dim).to(device)
        # Project and decode
        z_proj = model.decoder_input(z)
        z_proj = z_proj.view(z.size(0), *model.bottleneck_shape)
        samples = model.decoder(z_proj)
        
    save_image_grid(samples, "./results/figures/synthetic_samples.png", nrow=8, title="Synthetic Samples from Prior")
    print("Synthetic samples saved.")

    # 2. Latent Space Interpolation
    dataloader = get_dataloader(config)
    data, _ = next(iter(dataloader))
    data = data.to(device)
    
    # Get mu for two images
    with torch.no_grad():
        h = model.encoder(data[:2])
        h = h.view(h.size(0), -1)
        mu = model.fc_mu(h)
        
    mu1, mu2 = mu[0], mu[1]
    
    # Linear interpolation
    alphas = np.linspace(0, 1, 10)
    interpolations = []
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            z_interp = z_interp.unsqueeze(0)
            
            z_proj = model.decoder_input(z_interp)
            z_proj = z_proj.view(1, *model.bottleneck_shape)
            img = model.decoder(z_proj)
            interpolations.append(img)
            
    plot_interpolation(interpolations, "./results/figures/latent_interpolation.png")
    print("Interpolation plot saved.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/vae.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    generate_and_interpolate(args.config, args.checkpoint)