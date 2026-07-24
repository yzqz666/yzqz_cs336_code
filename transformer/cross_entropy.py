import torch
from transformer.softmax import softmax


def cross_entropy(inputs:torch.Tensor,targets:torch.Tensor):
    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_logits
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1)) + max_logits.squeeze(-1)
    correct_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = (log_sum_exp - correct_logits).mean()
    return loss

