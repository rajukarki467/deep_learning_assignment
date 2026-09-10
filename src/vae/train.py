import os
import sys
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.vae.model import VAE
from src.vae.loss import vae_loss
from src.vae.dataset import get_dataloader
from src.utils.seed import set_seed

def train(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config['seed'])
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    dataloader = get_dataloader(config)
    
    # Model
    model = VAE(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Create save dir
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    
    # Training Loop
    model.train()
    for epoch in range(config['training']['epochs']):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            
            loss, recon_loss, kl_loss = vae_loss(
                recon_batch, data, mu, logvar, 
                beta=config['model']['beta']
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            
            pbar.set_postfix({'Loss': loss.item(), 'Recon': recon_loss.item(), 'KL': kl_loss.item()})
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(config['training']['save_dir'], f"vae_epoch_{epoch+1}.pth"))
            
    # Save final model
    torch.save(model.state_dict(), os.path.join(config['training']['save_dir'], "vae_final.pth"))
    print("Training Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/vae.yaml', help='Path to config file')
    args = parser.parse_args()
    train(args.config)