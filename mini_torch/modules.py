# mini_torch/modules.py
import numpy as np
from .tensor import Tensor


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

class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        
        # Initialize weights: (out_channels, in_channels, kernel_h, kernel_w)
        k_h, k_w = self.kernel_size
        self.W = Tensor(np.random.randn(out_channels, in_channels, k_h, k_w) * 
                       np.sqrt(2.0 / (in_channels * k_h * k_w)))
        self.b = Tensor(np.zeros(out_channels))
    
    def forward(self, x):
        # x shape: (batch, in_channels, H, W)
        batch, in_c, H, W = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        
        # Calculate output dimensions
        out_h = (H - k_h) // s_h + 1
        out_w = (W - k_w) // s_w + 1
        
        # Create output tensor using im2col approach
        # This is more efficient and easier for backprop
        cols = self._im2col(x.data, k_h, k_w, s_h, s_w)
        # cols shape: (batch * out_h * out_w, in_channels * k_h * k_w)
        
        # Reshape weights for matrix multiplication
        W_col = self.W.data.reshape(self.out_channels, -1).T
        # W_col shape: (in_channels * k_h * k_w, out_channels)
        
        # Perform convolution as matrix multiplication
        out_data = np.matmul(cols, W_col) + self.b.data
        # out_data shape: (batch * out_h * out_w, out_channels)
        
        # Reshape to proper output format
        out_data = out_data.reshape(batch, out_h, out_w, self.out_channels)
        out_data = out_data.transpose(0, 3, 1, 2)
        # out_data shape: (batch, out_channels, out_h, out_w)
        
        out = Tensor(out_data, (x, self.W, self.b), op="conv2d")
        
        def _backward():
            # Gradient w.r.t output: (batch, out_channels, out_h, out_w)
            grad_out = out.grad.transpose(0, 2, 3, 1).reshape(batch * out_h * out_w, self.out_channels)
            
            # Gradient w.r.t weights
            grad_W = np.matmul(cols.T, grad_out)
            grad_W = grad_W.T.reshape(self.W.shape)
            self.W.grad += grad_W
            
            # Gradient w.r.t bias
            grad_b = grad_out.sum(axis=0)
            self.b.grad += grad_b
            
            # Gradient w.r.t input
            grad_cols = np.matmul(grad_out, W_col.T)
            grad_x = self._col2im(grad_cols, x.shape, k_h, k_w, s_h, s_w)
            x.grad += grad_x
        
        out._backward = _backward
        return out
    
    def _im2col(self, x, k_h, k_w, s_h, s_w):
        """Convert image to column matrix for efficient convolution."""
        batch, in_c, H, W = x.shape
        out_h = (H - k_h) // s_h + 1
        out_w = (W - k_w) // s_w + 1
        
        cols = np.zeros((batch, in_c, k_h, k_w, out_h, out_w))
        
        for y in range(k_h):
            y_max = y + s_h * out_h
            for x_idx in range(k_w):
                x_max = x_idx + s_w * out_w
                cols[:, :, y, x_idx, :, :] = x[:, :, y:y_max:s_h, x_idx:x_max:s_w]
        
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(batch * out_h * out_w, -1)
        return cols
    
    def _col2im(self, cols, x_shape, k_h, k_w, s_h, s_w):
        """Convert column matrix back to image for backpropagation."""
        batch, in_c, H, W = x_shape
        out_h = (H - k_h) // s_h + 1
        out_w = (W - k_w) // s_w + 1
        
        cols = cols.reshape(batch, out_h, out_w, in_c, k_h, k_w).transpose(0, 3, 4, 5, 1, 2)
        
        x_grad = np.zeros(x_shape)
        
        for y in range(k_h):
            y_max = y + s_h * out_h
            for x_idx in range(k_w):
                x_max = x_idx + s_w * out_w
                x_grad[:, :, y:y_max:s_h, x_idx:x_max:s_w] += cols[:, :, y, x_idx, :, :]
        
        return x_grad


class Conv2DTranspose(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        # Initialize weights: (in_channels, out_channels, kernel_h, kernel_w)
        k_h, k_w = self.kernel_size
        self.W = Tensor(np.random.randn(in_channels, out_channels, k_h, k_w) * 
                       np.sqrt(2.0 / (in_channels * k_h * k_w)))
        self.b = Tensor(np.zeros(out_channels))
    
    def forward(self, x):
        # x shape: (batch, in_channels, H, W)
        batch, in_c, H, W = x.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        
        # Calculate output dimensions before padding removal
        out_h = (H - 1) * s_h + k_h
        out_w = (W - 1) * s_w + k_w
        
        # Reshape input for efficient computation
        # (batch, in_channels, H, W) -> (batch * H * W, in_channels)
        x_reshaped = x.data.transpose(0, 2, 3, 1).reshape(batch * H * W, in_c)
        
        # Reshape weights for matrix multiplication
        # (in_channels, out_channels, k_h, k_w) -> (in_channels, out_channels * k_h * k_w)
        W_reshaped = self.W.data.reshape(in_c, -1)
        
        # Matrix multiplication: (batch*H*W, in_channels) @ (in_channels, out_channels*k_h*k_w)
        # Result: (batch*H*W, out_channels*k_h*k_w)
        cols = np.matmul(x_reshaped, W_reshaped)
        
        # Reshape to (batch, H, W, out_channels, k_h, k_w)
        cols = cols.reshape(batch, H, W, self.out_channels, k_h, k_w)
        
        # Use col2im to place the values in the output
        out_data = self._col2im_transpose(cols, batch, out_h, out_w, s_h, s_w)
        
        # Apply padding (crop edges)
        if p_h > 0 or p_w > 0:
            out_data = out_data[:, :, p_h:out_h-p_h, p_w:out_w-p_w]
        
        # Add bias
        out_data = out_data + self.b.data.reshape(1, -1, 1, 1)
        
        out = Tensor(out_data, (x, self.W, self.b), op="conv2d_transpose")
        
        def _backward():
            grad_out = out.grad
            
            # Restore padding for gradient computation
            if p_h > 0 or p_w > 0:
                grad_out_padded = np.zeros((batch, self.out_channels, out_h, out_w))
                grad_out_padded[:, :, p_h:out_h-p_h, p_w:out_w-p_w] = grad_out
                grad_out = grad_out_padded
            
            # Gradient w.r.t bias
            grad_b = grad_out.sum(axis=(0, 2, 3))
            self.b.grad += grad_b
            
            # Convert grad_out to columns using im2col
            grad_cols = self._im2col_transpose(grad_out, H, W, k_h, k_w, s_h, s_w)
            # grad_cols shape: (batch, H, W, out_channels, k_h, k_w)
            
            # Reshape for matrix operations
            # (batch, H, W, out_channels, k_h, k_w) -> (batch*H*W, out_channels*k_h*k_w)
            grad_cols_reshaped = grad_cols.reshape(batch * H * W, -1)
            
            # Gradient w.r.t weights
            # x_reshaped.T @ grad_cols_reshaped
            # (in_channels, batch*H*W) @ (batch*H*W, out_channels*k_h*k_w)
            # -> (in_channels, out_channels*k_h*k_w)
            grad_W = np.matmul(x_reshaped.T, grad_cols_reshaped)
            grad_W = grad_W.reshape(self.W.shape)
            self.W.grad += grad_W
            
            # Gradient w.r.t input
            # grad_cols_reshaped @ W_reshaped.T
            # (batch*H*W, out_channels*k_h*k_w) @ (out_channels*k_h*k_w, in_channels)
            # -> (batch*H*W, in_channels)
            grad_x_reshaped = np.matmul(grad_cols_reshaped, W_reshaped.T)
            grad_x = grad_x_reshaped.reshape(batch, H, W, in_c).transpose(0, 3, 1, 2)
            x.grad += grad_x
        
        out._backward = _backward
        return out
    
    def _col2im_transpose(self, cols, batch, out_h, out_w, s_h, s_w):
        """
        Convert columns to image for transposed convolution.
        cols shape: (batch, H, W, out_channels, k_h, k_w)
        Returns: (batch, out_channels, out_h, out_w)
        """
        _, H, W, out_channels, k_h, k_w = cols.shape
        out_data = np.zeros((batch, out_channels, out_h, out_w))
        
        for i in range(H):
            for j in range(W):
                h_start = i * s_h
                w_start = j * s_w
                h_end = h_start + k_h
                w_end = w_start + k_w
                
                # Add the kernel contribution to the output
                out_data[:, :, h_start:h_end, w_start:w_end] += cols[:, i, j, :, :, :]
        
        return out_data
    
    def _im2col_transpose(self, grad_out, H, W, k_h, k_w, s_h, s_w):
        """
        Extract columns from gradient for transposed convolution backward.
        grad_out shape: (batch, out_channels, out_h, out_w)
        Returns: (batch, H, W, out_channels, k_h, k_w)
        """
        batch, out_channels, out_h, out_w = grad_out.shape
        grad_cols = np.zeros((batch, H, W, out_channels, k_h, k_w))
        
        for i in range(H):
            for j in range(W):
                h_start = i * s_h
                w_start = j * s_w
                h_end = h_start + k_h
                w_end = w_start + k_w
                
                # Extract the kernel-sized region from grad_out
                grad_cols[:, i, j, :, :, :] = grad_out[:, :, h_start:h_end, w_start:w_end]
        
        return grad_cols
