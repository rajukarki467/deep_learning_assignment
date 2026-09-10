import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar, beta=1.0, loss_type='BCE'):
    """
    Computes the VAE loss: Reconstruction Loss + beta * KL Divergence.
    """
    if loss_type == 'BCE':
        # Binary Cross Entropy (sum over pixels, mean over batch)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif loss_type == 'MSE':
        # Mean Squared Error
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    else:
        raise ValueError("loss_type must be 'BCE' or 'MSE'")

    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Note: logvar is log(sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Normalize by batch size to keep loss scale reasonable
    batch_size = x.size(0)
    total_loss = (recon_loss + beta * kl_loss) / batch_size
    
    return total_loss, recon_loss / batch_size, kl_loss / batch_size