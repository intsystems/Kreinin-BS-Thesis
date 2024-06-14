import torch
import numpy as np

class OptimizerTemplate:
    def __init__(self, params, lr):
        self.params = list(params)
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
        for p in self.params:
            if p.grad is None: # We skip parameters without any gradients
                continue
            self.update_param(p)

    def update_param(self, p):
        # To be implemented in optimizer-specific classes
        raise NotImplementedError
    
    def get_param(self, p):
        raise NotImplemeteedError

class AdamW(OptimizerTemplate):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.param_step = {p: 0 for p in self.params} # Remembers "t" for each parameter for bias correction
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.g_t  = {p: torch.zeros_like(p.data) for p in self.params}
        self.D_t  = {p: torch.zeros_like(p.data) for p in self.params}
        self.grad = {p: torch.zeros_like(p.data) for p in self.params}
    def update_param(self, p):
        self.param_step[p] += 1
        
        self.grad[p] = torch.clone(p.grad)
        
        self.param_momentum[p]     = (1 - self.beta1) * p.grad + self.beta1 * self.param_momentum[p]
        self.param_2nd_momentum[p] = (1 - self.beta2) * (p.grad)**2 + self.beta2 * self.param_2nd_momentum[p]

        bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
        bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

        p_2nd_mom = self.param_2nd_momentum[p] / bias_correction_2
        p_mom     = self.param_momentum[p] / bias_correction_1

        self.D_t[p] = torch.sqrt(self.param_2nd_momentum[p] / bias_correction_2) + self.eps
        self.g_t[p] = self.param_momentum[p] / bias_correction_1

        p.add_(- self.lr * self.g_t[p] / self.D_t[p] - self.lr*self.weight_decay * p)
    
    def get_param(self, p):
        return {
            'grad'   : self.grad[p],
            'grad_L2': self.grad[p] + self.weight_decay*p,
            'g_adamw': self.grad[p] + self.weight_decay*p*(torch.sqrt(self.param_2nd_momentum[p]) + self.eps), 
            'D_t'    : self.D_t[p],
        }
    

class AdamL2(OptimizerTemplate):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.param_step = {p: 0 for p in self.params} # Remembers "t" for each parameter for bias correction
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.g_t = {p: torch.zeros_like(p.data) for p in self.params}
        self.D_t = {p: torch.zeros_like(p.data) for p in self.params}
        self.grad = {p: torch.zeros_like(p.data) for p in self.params}
        
    def update_param(self, p):
        self.param_step[p] += 1
        self.grad[p] = torch.clone(p.grad)
        grad = p.grad + self.weight_decay*p
        self.param_momentum[p] = (1 - self.beta1) * (grad) + self.beta1 * self.param_momentum[p]
        self.param_2nd_momentum[p] = (1 - self.beta2) * (grad)**2 + self.beta2 * self.param_2nd_momentum[p]

        bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
        bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

        self.D_t[p] = torch.sqrt(self.param_2nd_momentum[p] / bias_correction_2) + self.eps
        self.g_t[p] = self.param_momentum[p] / bias_correction_1

        p.add_(- self.lr * self.g_t[p] / self.D_t[p])
    
    def get_param(self, p):
        return {
            'grad'   : self.grad[p],
            'grad_L2': self.grad[p] + self.weight_decay*p,
            'g_adamw': self.grad[p] + self.weight_decay*p*(torch.sqrt(self.param_2nd_momentum[p]) + self.eps), 
            'D_t'    : self.D_t[p],
        }