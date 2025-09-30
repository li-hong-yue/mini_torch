# mini_torch/functions.py
import numpy as np
from .tensor import Tensor

def sigmoid(x: Tensor):
    """
    Functional sigmoid for your custom Tensor.
    Returns a new Tensor with autograd support.
    """
    out_data = 1 / (1 + np.exp(-x.data))
    out = Tensor(out_data, (x,), op="sigmoid")
    
    def _backward():
        x.grad += out.grad * out.data * (1 - out.data)
    
    out._backward = _backward
    return out

def relu(x: Tensor):
    out_data = np.maximum(0, x.data)
    out = Tensor(out_data, (x,), op="relu")
    
    def _backward():
        x.grad += out.grad * (x.data > 0).astype(x.data.dtype)
    
    out._backward = _backward
    return out

def tanh(x: Tensor):
    out_data = np.tanh(x.data)
    out = Tensor(out_data, (x,), op="tanh")
    
    def _backward():
        x.grad += out.grad * (1 - out.data ** 2)
    
    out._backward = _backward
    return out

def softmax(x: Tensor, axis=-1):
    e_x = np.exp(x.data - np.max(x.data, axis=axis, keepdims=True))
    out_data = e_x / np.sum(e_x, axis=axis, keepdims=True)
    out = Tensor(out_data, (x,), op="softmax")
    
    def _backward():
        # full Jacobian vector multiplication
        grad = out.grad
        sum_grad = np.sum(grad * out.data, axis=axis, keepdims=True)
        x.grad += (grad - sum_grad) * out.data
    
    out._backward = _backward
    return out

def cross_entropy(logits: Tensor, targets: np.ndarray):
    """
    logits: Tensor of shape (batch, num_classes)
    targets: np.ndarray of shape (batch,) with class indices
    """
    # Compute softmax
    e_x = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
    probs = e_x / np.sum(e_x, axis=1, keepdims=True)
    
    # Negative log likelihood
    batch_size = logits.data.shape[0]
    correct_logprobs = -np.log(probs[np.arange(batch_size), targets])
    loss_val = correct_logprobs.mean()
    
    # Wrap as Tensor
    loss = Tensor(loss_val, (logits,), op="cross_entropy")
    
    def _backward():
        grad = probs.copy()
        grad[np.arange(batch_size), targets] -= 1
        grad /= batch_size
        logits.grad += grad  # propagate to logits
    
    loss._backward = _backward
    return loss