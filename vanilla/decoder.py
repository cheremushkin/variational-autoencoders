import torch
from torch import nn
from pytorch_blocks import ConvTransposeBlock


class Decoder(nn.Module):
    """
    VAE Decoder
    """
    def __init__(self,
                 z_size: int,
                 n_features: int,
                 n_layers: int,
                 n_channels: int):
        """
        :param z_size: size of noise vector z
        :param n_features: size of the last feature map
        :param n_layers: number of layers in the generator
        :param n_channels: number of the output channels
        """
        super().__init__()

        self.z_size = z_size
        self.transform_input = nn.Linear(in_features=z_size, out_features=z_size * 4)

        blocks = nn.ModuleList()
        in_channels = z_size
        for n_layer in reversed(range(n_layers)):
            out_channels = n_features * (2 ** n_layer)
            blocks.append(ConvTransposeBlock(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=(3, 3),
                                             stride=2,
                                             padding=1,
                                             output_padding=1,
                                             bias=False))
            in_channels = out_channels
        self.layers = nn.Sequential(*blocks)
        self.activation = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=n_channels,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                output_padding=1,
                bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform_input(x).view(-1, self.z_size, 2, 2)
        x = self.layers(x)
        x = self.activation(x)
        return x
