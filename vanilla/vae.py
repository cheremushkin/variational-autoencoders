import typing as tp
import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class VAE(nn.Module):
    """
    Variational Auto Encoder
    """
    def __init__(self,
                 img_size: int,
                 n_channels: int,
                 n_features: int,
                 n_layers: int,
                 z_size: int):
        """
        :param img_size: original image size
        :param n_channels: number of the output channels
        :param n_features: base feature size of layers
        :param n_layers: number of layers in the generator
        :param z_size: dimension size of latent parameters
        """
        super().__init__()

        self.encoder = Encoder(img_size=img_size,
                               in_channels=n_channels,
                               n_features=n_features,
                               n_layers=n_layers,
                               z_size=z_size)
        self.decoder = Decoder(z_size=z_size,
                               n_features=n_features,
                               n_layers=n_layers,
                               n_channels=n_channels)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample, mu, log_var = self.encoder(x)
        x = self.decoder(sample)
        return x, mu, log_var
