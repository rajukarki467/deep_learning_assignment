import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=32, hidden_dims=[32, 64, 128, 256]):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # --- Encoder ---
        encoder_layers = []
        curr_channels = in_channels
        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(curr_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            curr_channels = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Dynamically compute the flattened size by passing a dummy tensor
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 32, 32)
            dummy_output = self.encoder(dummy_input)
            self.flatten_dim = dummy_output.view(1, -1).size(1)
        print(f"[VAE] Encoder output flattened dim: {self.flatten_dim}")
        
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # --- Decoder ---
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        # Compute the spatial dimensions at the bottleneck (e.g., 2x2 for 28x28 input)
        with torch.no_grad():
            self.bottleneck_shape = dummy_output.shape[1:]  # (C, H, W)
        
        decoder_layers = []
        reversed_hidden_dims = hidden_dims[::-1]
        for i in range(len(reversed_hidden_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(reversed_hidden_dims[i], reversed_hidden_dims[i+1], 
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(reversed_hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
            
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(reversed_hidden_dims[-1], out_channels=in_channels, 
                                   kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        z_projected = self.decoder_input(z)
        # Use the dynamically computed bottleneck shape
        z_projected = z_projected.view(z.size(0), *self.bottleneck_shape)
        recon_x = self.decoder(z_projected)
        
        return recon_x, mu, logvar