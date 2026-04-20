import torch
import torch.nn as nn
from einops import rearrange, einsum

def ScaledDotProductAttention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor = None) -> torch.Tensor:
    K_T = K.transpose(-2,-1)

    score = einsum(Q,K_T,"... queries d_k, ... d_k keys -> ... queries keys")
    d_k = Q.shape[-1]
    score = score / (d_k ** 0.5)
    if mask is not None:
        score = score.masked_fill(mask == 0, float("-inf"))

    attn_weights = torch.softmax(score, dim=-1)

    return einsum(attn_weights,V,"... queries keys, ... keys d_v -> ... queries d_v")
    
