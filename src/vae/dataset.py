import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloader(config):
    """Returns the dataloader for the specified dataset."""
    dataset_name = config['data']['dataset']
    batch_size = config['data']['batch_size']
    data_dir = config['data']['data_dir']
    
    transform = transforms.Compose([
        transforms.Resize(32),  
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)) # Optional: standardize
    ])
    
    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == "CelebA":
        # Note: CelebA requires manual download or specific torchvision setup
        dataset = datasets.CelebA(root=data_dir, split='train', download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config['data']['num_workers'])