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

# 1. batch - g, batch - D
# 2. batch - g, D - update by g (not \nabla f)
# 3. batch - but with beta = 0, batch - D
# 4. batch - but with beta = 0, update D - by g (not \nabla f)
# 5. AdamW - full AdamW
class AdamW_batch_batch(OptimizerTemplate):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8, params_percent=0.1):
        super().__init__(params, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.param_step = {p: 0 for p in self.params} # Remembers "t" for each parameter for bias correction
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.params_percent = params_percent

    def update_param(self, p):
        self.param_step[p] += 1

        perm = torch.randperm(p.grad.size(0))
        ids = perm[:int(len(p.grad) * self.params_percent)]
        # Update grad and D_t
        self.param_momentum[p][ids] = (1 - self.beta1) * p.grad[ids] + self.beta1 * self.param_momentum[p][ids]
        self.param_2nd_momentum[p][ids] = (1 - self.beta2) * (p.grad[ids])**2 + self.beta2 * self.param_2nd_momentum[p][ids]
        
        # Bias correction
        bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
        bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

        # Update preconditioning
        p_2nd_mom = self.param_2nd_momentum[p] / bias_correction_2
        p_mom = self.param_momentum[p] / bias_correction_1

        # Update weights
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr * p_mom - self.weight_decay * p
        p.add_(p_update)

class AdamW_batch_g(OptimizerTemplate):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8, params_percent=0.1):
        super().__init__(params, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.param_step = {p: 0 for p in self.params} # Remembers "t" for each parameter for bias correction
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.params_percent = params_percent

    def update_param(self, p):
        self.param_step[p] += 1

        perm = torch.randperm(p.grad.size(0))
        ids = perm[:int(len(p.grad) * self.params_percent)]
        # Update grad and D_t
        self.param_momentum[p][ids] = (1 - self.beta1) * p.grad[ids] + self.beta1 * self.param_momentum[p][ids]
        self.param_2nd_momentum[p] = (1 - self.beta2) * (self.param_momentum[p])**2 + self.beta2 * self.param_2nd_momentum[p]
        
        # Bias correction
        bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
        bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

        # Update preconditioning
        p_2nd_mom = self.param_2nd_momentum[p] / bias_correction_2
        p_mom = self.param_momentum[p] / bias_correction_1

        # Update weights
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr * p_mom - self.weight_decay * p
        p.add_(p_update)


class AdamW_sega_batch(OptimizerTemplate):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8, params_percent=0.1):
        super().__init__(params, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.param_step = {p: 0 for p in self.params} # Remembers "t" for each parameter for bias correction
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.params_percent = params_percent

    def update_param(self, p):
        self.param_step[p] += 1

        perm = torch.randperm(p.grad.size(0))
        ids = perm[:int(len(p.grad) * self.params_percent)]
        # Update grad and D_t
        self.param_momentum[p][ids] = p.grad[ids]
        self.param_2nd_momentum[p][ids] = (1 - self.beta2) * (p.grad[ids])**2 + self.beta2 * self.param_2nd_momentum[p][ids]
        
        # Bias correction
        bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
        bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

        # Update preconditioning
        p_2nd_mom = self.param_2nd_momentum[p] / bias_correction_2
        p_mom = self.param_momentum[p] / bias_correction_1

        # Update weights
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr * p_mom - self.weight_decay * p
        p.add_(p_update)

class AdamW_sega_g(OptimizerTemplate):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8, params_percent=0.1):
        super().__init__(params, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.param_step = {p: 0 for p in self.params} # Remembers "t" for each parameter for bias correction
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.params_percent = params_percent

    def update_param(self, p):
        self.param_step[p] += 1

        perm = torch.randperm(p.grad.size(0))
        ids = perm[:int(len(p.grad) * self.params_percent)]
        # Update grad and D_t
        self.param_momentum[p][ids] = p.grad[ids]
        self.param_2nd_momentum[p] = (1 - self.beta2) * (self.param_momentum[p])**2 + self.beta2 * self.param_2nd_momentum[p]
        
        # Bias correction
        bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
        bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

        # Update preconditioning
        p_2nd_mom = self.param_2nd_momentum[p] / bias_correction_2
        p_mom = self.param_momentum[p] / bias_correction_1

        # Update weights
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr * p_mom - self.weight_decay * p
        p.add_(p_update)

class AdamW(OptimizerTemplate):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8, params_percent=0.1):
        super().__init__(params, lr)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.param_step = {p: 0 for p in self.params} # Remembers "t" for each parameter for bias correction
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}
        self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}

    def update_param(self, p):
        self.param_step[p] += 1
        
        self.param_momentum[p] = (1 - self.beta1) * p.grad + self.beta1 * self.param_momentum[p]
        self.param_2nd_momentum[p] = (1 - self.beta2) * (p.grad)**2 + self.beta2 * self.param_2nd_momentum[p]

        bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
        bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

        p_2nd_mom = self.param_2nd_momentum[p] / bias_correction_2
        p_mom = self.param_momentum[p] / bias_correction_1
        p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
        p_update = -p_lr * p_mom - self.weight_decay * p

        p.add_(p_update)
