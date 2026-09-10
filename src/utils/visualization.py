import matplotlib.pyplot as plt 
import torch 
import numpy as np 

def save_image_grid(images,filename,nrow=8,title=None):
    """
    Save a grid of images to a file.

    Args:
        images (torch.Tensor): A tensor of shape (N, C, H, W) containing the images.
        filename (str): The path to save the image grid.
        nrow (int): Number of images in each row of the grid. Default is 8.
        title (str): Optional title for the image grid. Default is None.
    """
    images = images.detach().cpu()
    if images.shape[1] == 1: # Grayscale
        images = images.squeeze(1)
    
    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow, nrow))
    if title:
        fig.suptitle(title)
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            ax.imshow(images[i], cmap='gray' if images.shape[1] == 1 else None)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_reconstruction(original, reconstructed, filename, n=8):
    """Plots original vs reconstructed images."""
    orig = original[:n].detach().cpu()
    recon = reconstructed[:n].detach().cpu()
    
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(orig[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    
    axes[0, 0].set_title("Original", loc='left')
    axes[1, 0].set_title("Reconstructed", loc='left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_interpolation(interpolations, filename):
    """Plots a linear interpolation sequence."""
    fig, axes = plt.subplots(1, len(interpolations), figsize=(len(interpolations) * 1.5, 2))
    for i, img in enumerate(interpolations):
        axes[i].imshow(img.squeeze().detach().cpu(), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()