from transformer.transformer_block import TransformerBlock
import torch
import torch.nn as nn

from transformer.softmax import softmax
from transformer.Linear import Linear
from transformer.RMSNorm import RMSNorm
from transformer.Embedding import Embedding


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output_linear = Linear(d_model, vocab_size)
        self.embedding = Embedding(vocab_size, d_model)
        
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        x = self.embedding.forward(x)
        for block in self.transformer_blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        output = self.output_linear.forward(x)
        print(output.shape)
        return output