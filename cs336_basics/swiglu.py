import torch
import torch.nn as nn
from .linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)   
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.SiLU(self.linear1(x)) * self.linear3(x))

    def SiLU(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
