import torch
import torch.nn as nn
from einops import rearrange, einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, 'd_k must be even'
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        dim = torch.arange(0, d_k, 2).float()
        dim = theta ** (dim / d_k)
        dim_inv = 1 / dim
        position_ids = torch.arange(max_seq_len).float()
        freqs = torch.outer(position_ids, dim_inv)
        
        sin = freqs.sin()
        cos = freqs.cos()

        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('cos', cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_odd * cos + x_even * sin

        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_even_rotated
        x_rotated[..., 1::2] = x_odd_rotated

        return x_rotated



    