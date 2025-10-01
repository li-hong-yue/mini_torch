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


def binary_cross_entropy(pred: Tensor, target: Tensor, epsilon=1e-8, reduction='sum'):
    """
    Binary cross entropy loss.
    BCE = -[target*log(pred) + (1-target)*log(1-pred)]
    
    pred: Tensor of predicted probabilities (should be in [0, 1])
    target: Tensor of ground truth (should be in [0, 1])
    epsilon: small value to avoid log(0)
    """
    # Clip predictions to avoid log(0)
    pred_clipped = np.clip(pred.data, epsilon, 1 - epsilon)
    
    # Compute BCE
    bce_data = -(target.data * np.log(pred_clipped) + 
                 (1 - target.data) * np.log(1 - pred_clipped))
    if reduction == 'sum':
        loss_val = bce_data.sum()
    else:
        assert reduction == 'mean'
        loss_val = bce_data.mean()
    
    out = Tensor(loss_val, (pred, target), op="bce")
    
    def _backward():
        # Gradient w.r.t pred: -(target/pred - (1-target)/(1-pred)) / batch_size
        batch_size = np.prod(pred.data.shape)
        grad_pred = -(target.data / pred_clipped - (1 - target.data) / (1 - pred_clipped))
        pred.grad += grad_pred * out.grad #/ batch_size
        
        # Gradient w.r.t target (usually not needed, but for completeness)
        grad_target = -np.log(pred_clipped) + np.log(1 - pred_clipped)

        if reduction == 'sum':
            pred.grad += grad_pred * out.grad 
            target.grad += grad_target * out.grad 
        else:
            assert reduction == 'mean'
            pred.grad += grad_pred * out.grad / batch_size
            target.grad += grad_target * out.grad / batch_size
    
    out._backward = _backward
    return out

def sum(x: Tensor, axis=None, keepdims=False):
    """
    Sum reduction along specified axis.
    
    x: Tensor to sum
    axis: axis or axes to sum over (None means sum all)
    keepdims: whether to keep the reduced dimensions
    """
    out_data = np.sum(x.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, (x,), op="sum")
    
    def _backward():
        if axis is None:
            # Sum over all dimensions - broadcast to original shape
            x.grad += np.ones_like(x.data) * out.grad
        else:
            # Sum over specific axis - need to broadcast gradient back
            if keepdims:
                x.grad += np.broadcast_to(out.grad, x.shape)
            else:
                # Expand dims to match original shape
                grad_expanded = np.expand_dims(out.grad, axis=axis)
                x.grad += np.broadcast_to(grad_expanded, x.shape)
    
    out._backward = _backward
    return out