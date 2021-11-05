import typing as tp
import torch
import torch.nn as nn
from pytorch_blocks import ConvBlock


class Encoder(nn.Module):
    """
    VAE Encoder
    """
    def __init__(self,
                 img_size: int,
                 in_channels: int,
                 n_features: int,
                 n_layers: int,
                 z_size: int):
        """
        :param img_size: original image size
        :param in_channels: number of the output channels
        :param n_features: base feature size of layers
        :param n_layers: number of layers in the generator
        :param z_size: dimension size of latent parameters
        """
        super().__init__()

        blocks = nn.ModuleList()
        for n_layer in range(n_layers):
            out_channels = n_features * (2 ** n_layer)
            blocks.append(ConvBlock(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=(3, 3),
                                    stride=2,
                                    padding=1,
                                    bias=False))
            in_channels = out_channels

        self.layers = nn.Sequential(*blocks)
        self.mu = nn.Linear(in_features=in_channels * (img_size // 2 ** n_layers) ** 2, out_features=z_size)
        self.log_var = nn.Linear(in_features=in_channels * (img_size // 2 ** n_layers) ** 2, out_features=z_size)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.layers(x).flatten(start_dim=1)

        mu = self.mu(x)
        log_var = self.log_var(x)
        sample = self.reparameterize(mu, log_var)
        return sample, mu, log_var
