import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module): 
    def __init__(self,in_features:int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device 
        self.dtype = dtype
        self.w = nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        std = 2/(in_features + out_features)**0.5
        nn.init.trunc_normal_(self.w, mean = 0,std = std,a = -3 * std,b = 3 * std)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        #  y = x @ self.w.T
        self.w.t = rearrange(self.w,"d_in d_out -> d_out d_in")
        y = einsum(x,self.w.t,"... d_in, d_in d_out -> ... d_out")
        return y
