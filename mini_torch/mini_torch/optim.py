# mini_torch/optim.py
import numpy as np


class SGD:
    def __init__(self, parameters, lr=1e-3):
        """
        parameters: list of Tensor objects (weights/biases)
        lr: learning rate
        """
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad  # gradient descent

    def zero_grad(self):
        for p in self.parameters:
            p.grad[:] = 0  # reset gradients

class Adam(SGD):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0  # timestep
        # Initialize moment estimates for each parameter
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self.t += 1
        beta1, beta2 = self.betas

        for i, p in enumerate(self.parameters):
            g = p.grad

            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            # Update biased second raw moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (g * g)

            # Bias correction
            m_hat = self.m[i] / (1 - beta1**self.t)
            v_hat = self.v[i] / (1 - beta2**self.t)

            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
