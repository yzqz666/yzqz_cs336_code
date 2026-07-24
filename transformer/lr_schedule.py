import math

class lr_schedule:
    def __init__(self,max_lr,min_lr,T_w,T_c):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_w = T_w
        self.T_c = T_c

    def get_lr(self,t):
        if t < self.T_w :
            return t / self.T_w * self.max_lr
        if self.T_w <= t and t < self.T_c:
            return self.min_lr + 0.5 * (1 + math.cos((t - self.T_w) / (self.T_c - self.T_w) * math.pi)) * (self.max_lr - self.min_lr)
        return self.min_lr