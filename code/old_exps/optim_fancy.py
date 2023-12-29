import torch
import numpy as np

class OptimizerTemplate:
    def __init__(self, params, lr, order = []):
        self.params = list(params)
        self.special_order = [x for x in order]
        self.lr = lr

    def zero_grad(self):
        ## Set gradients of all parameters to zero
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_() # For second-order optimizers important
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        ## Apply update step to all parameters
        i = 0
        for p in self.params:
            if p.grad is None: # We skip parameters without any gradients
                continue
            self.update_param(p, self.special_order[i])
            i +=1

    def update_param(self, p):
        # To be implemented in optimizer-specific classes
        raise NotImplementedError


class AdamW_fancy(OptimizerTemplate):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8, params_percent=0.1, order=[]):
        super().__init__(params, lr, order)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.param_step = {p: 0 for p in self.params} # Remembers "t" for each parameter for bias correction
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}

    def update_param(self, p, flag=False):
        self.param_step[p] += 1
        if flag:
            self.param_momentum[p] = (1 - self.beta1) * p.grad + self.beta1 * self.param_momentum[p]
            bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
            p_mom = self.param_momentum[p]# / bias_correction_1
            p_lr = self.lr
            p_update = -p_lr * p_mom - self.weight_decay * p
            p.add_(p_update)
        else:
            self.param_momentum[p] = (1 - self.beta1) * p.grad + self.beta1 * self.param_momentum[p]
            self.param_2nd_momentum[p] = (1 - self.beta2) * (p.grad)**2 + self.beta2 * self.param_2nd_momentum[p]

            bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
            bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

            p_2nd_mom = self.param_2nd_momentum[p] / bias_correction_2
            p_mom = self.param_momentum[p] / bias_correction_1
            p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
            p_update = -p_lr * p_mom - self.weight_decay * p
            p.add_(p_update)
