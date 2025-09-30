# mini_torch/tensor.py
import numpy as np

def _grad_broadcast(grad, shape):
    """
    Adjust grad to match the original tensor's shape after broadcasting.
    
    grad : np.ndarray
        Gradient coming from the output (may be broadcasted)
    shape : tuple
        Original shape of the tensor
    
    Returns:
        np.ndarray with shape == shape
    """
    grad_out = grad
    
    # Sum over extra leading dimensions
    while grad_out.ndim > len(shape):
        grad_out = grad_out.sum(axis=0)
    
    return grad_out

class Tensor:
    def __init__(self, data, _children=(), op=""):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
    
    @property
    def T(self):
        out = Tensor(self.data.T, (self,), op="T")
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out

    @property
    def shape(self):
        return self.data.shape
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), op="reshape")
        def _backward():
            # Gradient just reshaped back to original shape
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), op="+")
        
        def _backward():
            self.grad += _grad_broadcast(out.grad, self.data.shape)
            other.grad += _grad_broadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Tensor(-self.data, (self,), op="neg")
        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), op="*")
        
        def _backward():
            self.grad += _grad_broadcast(out.grad * other.data, self.data.shape)
            other.grad += _grad_broadcast(out.grad * self.data, other.data.shape)
    
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), op="/")
        
        def _backward():
            self.grad += _grad_broadcast(out.grad / other.data, self.data.shape)
            other.grad += _grad_broadcast(-out.grad * self.data / (other.data ** 2), other.data.shape)
        
        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), op="@")
        
        def _backward():
            grad_self = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
            grad_other = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
            self.grad += _grad_broadcast(grad_self, self.data.shape)
            other.grad += _grad_broadcast(grad_other, other.data.shape)
        
        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(self.data.mean(), (self,), op="mean")
        def _backward():
            self.grad += out.grad * np.ones_like(self.data) / self.data.size
        out._backward = _backward
        return out

    def pow(self, power):
        out = Tensor(self.data ** power, (self,), op=f"**{power}")
        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # topological order all the children of this node
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # seed gradient
        self.grad = np.ones_like(self.data)

        # go backwards
        for v in reversed(topo):
            v._backward()
