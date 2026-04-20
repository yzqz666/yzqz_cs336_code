import torch
import torch.nn as nn
from einops import rearrange, einsum
from transformer.multihead_self_attention import MultiHeadSelfAttention
from transformer.RoPE import rope
from transformer.RMSNorm import RMSNorm
from transformer.SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)

        self.attention.rope = rope(theta, d_model // num_heads, max_seq_len)
        self.norm = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        RMSNorm1x = self.norm.forward(x)
        attention_output = self.attention.forward(RMSNorm1x)
        middle_output = x + attention_output
        RMSNorm2x = self.norm2.forward(middle_output)
        ffn_output = self.ffn.forward(RMSNorm2x)
        output = middle_output + ffn_output
        return output