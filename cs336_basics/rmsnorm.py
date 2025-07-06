import torch
import torch.nn as nn
from einops import rearrange, einsum

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.randn(d_model, device=device, dtype=dtype)) # learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = self.RMS(x)
        rms_inv = 1 / rms
        x = einsum(x, rms_inv, '... seq_len d_model, ... seq_len -> ... seq_len d_model')
        x = einsum(x, self.g, '... seq_len d_model, d_model -> ... seq_len d_model')
        return x
        
    def RMS(self, x: torch.Tensor): 
        return torch.sqrt(torch.mean(x ** 2, dim=2) + self.eps)