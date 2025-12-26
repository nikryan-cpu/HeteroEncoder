import torch
import torch.nn as nn


class HeteroEncoderCVAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, latent_dim=64, desc_dim=3):
        super(HeteroEncoderCVAE, self).__init__()

        self.latent_dim = latent_dim

        # --- ENCODER ---
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        # Dense layers for Latent Space
        self.fc_mu = nn.Linear(hidden_dim + desc_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + desc_dim, latent_dim)

        # --- DECODER ---
        # Вход: Z + Energy (1 dim)
        self.fc_z_to_hidden = nn.Linear(latent_dim + 1, hidden_dim)
        self.decoder_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, descriptors, energy):
        # 1. Encode
        embedded = self.embedding(x)
        _, h_n = self.encoder_gru(embedded)
        h_n = h_n.squeeze(0)

        # Combine Text features + Descriptors
        combined = torch.cat([h_n, descriptors], dim=1)

        mu = self.fc_mu(combined)
        logvar = self.fc_logvar(combined)
        z = self.reparameterize(mu, logvar)

        # 2. Decode
        z_cond = torch.cat([z, energy], dim=1)
        hidden = self.fc_z_to_hidden(z_cond).unsqueeze(0)

        embedded_dec = self.embedding(x)
        output, _ = self.decoder_gru(embedded_dec, hidden)
        logits = self.fc_out(output)

        return logits, mu, logvar

    def sample_z(self, batch_size, device):
        return torch.randn(batch_size, self.latent_dim).to(device)