import torch

def gradient_clipping(parameters,max_l2_norm):
    tot = 0
    for p in parameters:
        if p.grad is None:
            continue
        tot += torch.sum(p.grad ** 2)

    tot_norm = torch.sqrt(tot)

    if tot_norm > max_l2_norm:
        scale = max_l2_norm / (1e-6 + tot_norm)
        for p in parameters :
            if p.grad is None :
                continue
            p.grad = p.grad * scale