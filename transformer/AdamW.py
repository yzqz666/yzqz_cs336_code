import torch 

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr,betas,eps,weight_decay):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None :
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                beta1,beta2 = group["betas"]
                state["step"] += 1
                state["m"] = state["m"] * beta1 + (1 - beta1) * grad
                state["v"] = state["v"] * beta2 + (1 - beta2) * grad * grad

                n_lr = group["lr"] * (1 - pow(beta2,state["step"])) ** 0.5 / (1 - pow(beta1,state["step"]))
                
                p.data = p.data - n_lr * state["m"] / (torch.sqrt(state["v"]) + group["eps"])
                p.data = (1 - group["lr"] * group["weight_decay"]) * p.data
