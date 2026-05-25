"""
Convolutional Neural Network (simplified implementation).

Mathematical Foundation:
- Convolution: out[i,j] = Σ(kernel * input[i:i+k_h, j:j+k_w]) + bias
- Max pooling: out = max(window)
- Flattening: reshape to 1D for FC layer
"""

import numpy as np
from typing import List, Tuple
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')


class ConvLayer:
    """
    Convolutional Layer.
    
    Attributes:
        n_filters: Number of filters
        kernel_size: Size of convolution kernel
        stride: Stride of convolution
        padding: Padding size
        kernels: Convolutional kernels
    """
    
    def __init__(self, n_filters: int, kernel_size: int = 3, stride: int = 1, 
                 padding: int = 0):
        """
        Initialize convolutional layer.
        
        Args:
            n_filters: Number of filters
            kernel_size: Kernel size (default: 3)
            stride: Stride (default: 1)
            padding: Padding (default: 0)
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize kernels randomly
        self.kernels = np.random.randn(n_filters, kernel_size, kernel_size) * 0.01
        self.bias = np.zeros(n_filters)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            X: Input (batch_size, height, width, channels) or (height, width)
            
        Returns:
            Convolved output
        """
        if X.ndim == 2:
            X = X[np.newaxis, :, :, np.newaxis]
        
        batch_size, height, width, channels = X.shape
        
        # Calculate output dimensions
        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Pad input
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding), (0, 0)))
        else:
            X_padded = X
        
        # Convolve
        output = np.zeros((batch_size, out_h, out_w, self.n_filters))
        
        for f in range(self.n_filters):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size
                    
                    # Extract window
                    window = X_padded[:, h_start:h_end, w_start:w_end, :]
                    
                    # Convolve with kernel
                    conv = np.zeros(batch_size)
                    for b in range(batch_size):
                        for c in range(channels):
                            conv[b] += np.sum(window[b, :, :, c] * self.kernels[f])
                    
                    output[:, i, j, f] = conv + self.bias[f]
        
        return output


class PoolingLayer:
    """
    Max Pooling Layer.
    
    Attributes:
        pool_size: Size of pooling window
        stride: Stride of pooling
    """
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        """
        Initialize pooling layer.
        
        Args:
            pool_size: Pool window size (default: 2)
            stride: Stride (default: 2)
        """
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass (max pooling).
        
        Args:
            X: Input
            
        Returns:
            Pooled output
        """
        batch_size, height, width, channels = X.shape
        
        # Calculate output dimensions
        out_h = (height - self.pool_size) // self.stride + 1
        out_w = (width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, out_h, out_w, channels))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                # Max pool
                window = X[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.max(window.reshape(batch_size, -1, channels), axis=1)
        
        return output


class SimpleCNN:
    """
    Simplified Convolutional Neural Network.
    
    Attributes:
        layers: List of layers
        output_dim: Output dimension for classification
    """
    
    def __init__(self, output_dim: int = 10):
        """
        Initialize SimpleCNN.
        
        Args:
            output_dim: Output dimension (default: 10 for MNIST)
        """
        self.output_dim = output_dim
        self.conv1 = ConvLayer(n_filters=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = PoolingLayer(pool_size=2, stride=2)
        self.conv2 = ConvLayer(n_filters=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = PoolingLayer(pool_size=2, stride=2)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through CNN.
        
        Args:
            X: Input images
            
        Returns:
            Feature maps
        """
        # First conv + pool block
        x = self.conv1.forward(X)
        x = np.maximum(x, 0)  # ReLU
        x = self.pool1.forward(x)
        
        # Second conv + pool block
        x = self.conv2.forward(x)
        x = np.maximum(x, 0)  # ReLU
        x = self.pool2.forward(x)
        
        # Flatten
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        return x_flat
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from input.
        
        Args:
            X: Input images
            
        Returns:
            Feature vector
        """
        return self.forward(X)
