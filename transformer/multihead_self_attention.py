import torch
import torch.nn as nn
from einops import rearrange, einsum
from transformer.Linear import Linear
from transformer.scaled_dot_product_attention import ScaledDotProductAttention
from transformer.RoPE import rope


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_k
        self.q_linear = Linear(d_model,d_model)
        self.k_linear = Linear(d_model,d_model)
        self.v_linear = Linear(d_model,d_model)
        self.output_linear = Linear(d_model,d_model)
        self.rope = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_linear.forward(x)
        k = self.k_linear.forward(x)
        v = self.v_linear.forward(x)

        q = rearrange(q, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        k = rearrange(k, "b s (h d_k) -> b h s d_k", h=self.num_heads)
        v = rearrange(v, "b s (h d_v) -> b h s d_v", h=self.num_heads)

        if self.rope is not None:
            token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        attention_output = ScaledDotProductAttention(q, k, v, mask)
        attention_output = rearrange(attention_output, "b h s d_v -> b s (h d_v)")
        output = self.output_linear.forward(attention_output)
       
        return output
        
