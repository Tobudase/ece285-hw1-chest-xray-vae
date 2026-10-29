"""Convolutional variational autoencoder for 64x64 grayscale radiographs."""

import torch
from torch import nn
from torch.nn import functional as F


class ChestVAE(nn.Module):
    """Encode radiographs into a Gaussian latent space and decode samples."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Flatten(),
        )
        self.mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.project = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        """Return the mean and log variance of q(z | x)."""
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        """Draw a latent sample while keeping gradients through the encoder."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        """Map latent vectors to grayscale images in [0, 1]."""
        return self.decoder(self.project(z).reshape(-1, 128, 4, 4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device, generator=None):
        """Generate new images from the standard Gaussian prior."""
        z = torch.randn(n, self.latent_dim, device=device, generator=generator)
        return self.decode(z)


def elbo_components(reconstruction, x, mu, logvar, beta=1.0):
    """Return batch means of BCE, KL, and negative beta-ELBO in nats/image."""
    # Sum over pixels and latent dimensions, then average over the batch.
    reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum") / x.size(0)
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / x.size(0)
    return reconstruction_loss, kl, reconstruction_loss + beta * kl
