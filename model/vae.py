import numpy as np
import torch
import torch.nn as nn
from typing import List
from torch import Tensor


def lengths_to_mask(
    lengths: List[int], device: torch.device, max_len: int = None
) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        device,
        nfeats,
        latent_dim=64,
        num_heads=1024,
        ff_size=8,
        dropout=0.1,
        num_layers=8,
        activation="relu",
    ):
        super(KLAutoEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.sequence_pos_decoder = PositionalEncoding(latent_dim, dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )

        self.Encoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=num_layers
        )
        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
        )
        self.Decoder = nn.TransformerDecoder(
            seqTransDecoderLayer, num_layers=num_layers
        )
        self.device = device
        self.skel_embedding = nn.Linear(nfeats, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, nfeats)
        self.dist_layer = nn.Linear(self.latent_dim, 2 * self.latent_dim)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        lengths = [len(feature) for feature in x]
        z, mean, log_var = self.encode(x, lengths)
        x_hat = self.decode(z, lengths)
        return x_hat, mean, log_var

    def encode(self, features, lengths=None):
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device
        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        x = self.skel_embedding(x)

        x = x.permute(1, 0, 2)  # nframes, bs, latent_dim

        x = self.sequence_pos_encoder(x)
        dist = self.Encoder(x, src_key_padding_mask=~mask)

        dist = self.dist_layer(dist)
        mu = dist[:, :, 0 : self.latent_dim]
        logvar = dist[:, :, self.latent_dim :]

        z = self.reparameterization(
            mu, torch.exp(0.5 * logvar)
        )  # takes exponential function (log var -> var)

        return z, mu, logvar

    def decode(self, z, lengths):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        queries = self.sequence_pos_decoder(queries)
        output = self.Decoder(
            tgt=queries,
            memory=z,
            tgt_key_padding_mask=~mask,
        )

        output = self.final_layer(output)
        output[~mask.T] = 0

        feats = output.permute(1, 0, 2)
        return feats


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)
