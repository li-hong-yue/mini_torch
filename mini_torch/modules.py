# mini_torch/modules.py
import numpy as np
from .tensor import Tensor
from .functions import sigmoid


class Module:
    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def __setattr__(self, name, value):
        # If value is a Module, register it automatically
        if isinstance(value, Module):
            self._modules[name] = value
        # If value is a Tensor, register it as a parameter
        elif isinstance(value, Tensor):
            self._parameters[name] = value
        # Always set the attribute normally
        super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        # Collect parameters recursively
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__() 
        self.W = Tensor(np.random.randn(in_features, out_features) * 0.01)
        self.b = Tensor(np.zeros(out_features))
    
    def forward(self, x):
        return x @ self.W + self.b