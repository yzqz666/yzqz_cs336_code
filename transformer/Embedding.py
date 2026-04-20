import torch
import torch.nn as nn
from einops import rearrange, einsum

class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device 
        self.dtype = dtype
        std = 1 ** 0.5
        self.embedding_matrix = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding_matrix, mean = 0,std = std,a = -3 ,b = 3)

    def forward(self, token_ids:torch.LongTensor) -> torch.Tensor:
        # output = einsum(token_ids,self.embedding_matrix,"... -> ... d_emb, d_emb")
        output = self.embedding_matrix[token_ids]
        return output
        