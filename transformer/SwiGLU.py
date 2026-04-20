import torch
import torch.nn as nn
from transformer.Linear import Linear

class SwiGLU(nn.Module):
    def __init__(self,d_model:int, d_ff = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.device = device 
        self.dtype = dtype
        self.d_ff = d_ff
        if d_ff is None:
            self.d_ff = d_model * 8 // 3
            self.d_ff = self.d_ff // 64 * 64
        self.linear1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        input1 = self.linear1.forward(x)
        input2 = self.linear3.forward(x)
        hidden = input1 * torch.sigmoid(input1) * input2
        output = self.linear2.forward(hidden)
        return output