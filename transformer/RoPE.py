import torch
import torch.nn as nn
from einops import rearrange, einsum

class rope(nn.Module):
    def __init__(self,theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / self.d_k))
        position = torch.arange(max_seq_len, device=device)
        sinusoids = einsum(position,freq,"n,q -> n q")

        self.register_buffer("cos_",sinusoids.cos(),persistent=False)
        self.register_buffer("sin_",sinusoids.sin(),persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_[token_positions]
        sin = self.sin_[token_positions]

        x_part1 = x[..., 0::2]
        x_part2 = x[..., 1::2]

        output1 = x_part1 * cos - x_part2 * sin 
        output2 = x_part1 * sin + x_part2 * cos

        
        out = torch.stack([output1, output2], dim=-1)
        out = out.flatten(-2)
        return out