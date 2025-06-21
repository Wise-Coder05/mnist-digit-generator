import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(10, 10)

        self.encoder = nn.Sequential(
            nn.Linear(28*28 + 10, 400),
            nn.ReLU()
        )
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 10, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x, y):
        y_embed = self.label_embedding(y)
        x = torch.cat([x.view(-1, 784), y_embed], dim=1)
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        y_embed = self.label_embedding(y)
        z = torch.cat([z, y_embed], dim=1)
        return self.decoder(z)
