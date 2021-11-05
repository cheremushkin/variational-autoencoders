import torch
from torch import nn


class ELBO(nn.Module):
    def __init__(self,
                 reconstruction: nn.Module):
        super().__init__()
        self.reconstruction = reconstruction

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor,
                mu: torch.Tensor,
                log_var: torch.Tensor) -> torch.autograd.Variable:
        reconstruction = self.reconstruction(input, target)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl + reconstruction
