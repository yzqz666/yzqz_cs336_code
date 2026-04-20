"""
RMSNorm -> Root Mean Square Layer Normalization
"""
import torch
import torch.nn as nn
from einops import rearrange, einsum

class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        input_shape = x.dtype
        x = x.to(torch.float32)
        norm_x = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm_x = norm_x.to(input_shape)
        return norm_x * self.weight
        
