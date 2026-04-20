import torch
import torch.nn as nn
from einops import rearrange, einsum

from transformer.softmax import softmax

x = torch.tensor([[1, 2, 3], 
                  [4, 5, 6]])
print(x.shape)
softmax(x, dim=1)


