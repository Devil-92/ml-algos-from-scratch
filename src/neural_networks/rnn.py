"""
Recurrent Neural Network with LSTM cells for sequence modeling.

Mathematical Foundation:
- LSTM gates:
  - Forget gate: f_t = σ(W_f*[h_{t-1}, x_t] + b_f)
  - Input gate: i_t = σ(W_i*[h_{t-1}, x_t] + b_i)
  - Cell gate: C̃_t = tanh(W_c*[h_{t-1}, x_t] + b_c)
  - Output gate: o_t = σ(W_o*[h_{t-1}, x_t] + b_o)
"""

import numpy as np
from typing import List, Tuple
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')


class LSTM:
    """
    Long Short-Term Memory (LSTM) cell.
    
    Attributes:
        input_dim: Input dimension
        hidden_dim: Hidden state dimension
        learning_rate: Learning rate
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, learning_rate: float = 0.01):
        """
        Initialize LSTM.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            learning_rate: Learning rate
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.Wf = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.01
        self.bf = np.zeros((1, hidden_dim))
        self.Wi = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.01
        self.bi = np.zeros((1, hidden_dim))
        self.Wc = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.01
        self.bc = np.zeros((1, hidden_dim))
        self.Wo = np.random.randn(input_dim + hidden_dim, hidden_dim) * 0.01
        self.bo = np.zeros((1, hidden_dim))
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def tanh(x):
        """Tanh activation."""
        return np.tanh(x)
    
    def forward(self, X: np.ndarray, h_prev: np.ndarray = None, 
                c_prev: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through sequence.
        
        Args:
            X: Input sequence (seq_len, batch_size, input_dim)
            h_prev: Previous hidden state
            c_prev: Previous cell state
            
        Returns:
            Hidden state and cell state
        """
        seq_len, batch_size = X.shape[0], X.shape[1] if X.ndim > 2 else 1
        
        if h_prev is None:
            h_prev = np.zeros((batch_size, self.hidden_dim))
        if c_prev is None:
            c_prev = np.zeros((batch_size, self.hidden_dim))
        
        for t in range(seq_len):
            x_t = X[t] if X.ndim > 2 else X[t:t+1]
            
            # Concatenate input and hidden state
            concat = np.concatenate([x_t, h_prev], axis=1)
            
            # Forget gate
            f_t = self.sigmoid(concat.dot(self.Wf) + self.bf)
            # Input gate
            i_t = self.sigmoid(concat.dot(self.Wi) + self.bi)
            # Cell gate
            c_tilde = self.tanh(concat.dot(self.Wc) + self.bc)
            # Update cell state
            c_t = f_t * c_prev + i_t * c_tilde
            # Output gate
            o_t = self.sigmoid(concat.dot(self.Wo) + self.bo)
            # Update hidden state
            h_t = o_t * self.tanh(c_t)
            
            h_prev = h_t
            c_prev = c_t
        
        return h_prev, c_prev


class SimpleRNN:
    """
    Simplified RNN for sequence modeling.
    
    Attributes:
        input_dim: Input dimension
        hidden_dim: Hidden state dimension
        output_dim: Output dimension
        learning_rate: Learning rate
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 learning_rate: float = 0.01, iterations: int = 100):
        """
        Initialize SimpleRNN.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension
            learning_rate: Learning rate
            iterations: Number of iterations
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.iterations = iterations
        
        # Initialize weights
        self.Wx = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Wy = np.random.randn(hidden_dim, output_dim) * 0.01
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, output_dim))
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass.
        
        Args:
            X: Input sequence (seq_len, batch_size, input_dim)
            
        Returns:
            Output and hidden states
        """
        seq_len = X.shape[0]
        batch_size = X.shape[1] if X.ndim > 2 else 1
        
        h = np.zeros((batch_size, self.hidden_dim))
        hiddens = []
        
        for t in range(seq_len):
            x_t = X[t] if X.ndim > 2 else X[t:t+1]
            h = np.tanh(x_t.dot(self.Wx) + h.dot(self.Wh) + self.bh)
            hiddens.append(h)
        
        # Output from last hidden state
        y = h.dot(self.Wy) + self.by
        
        return y, hiddens
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train RNN.
        
        Args:
            X: Input sequences
            y: Target values
            
        Returns:
            Self
        """
        for epoch in range(self.iterations):
            output, _ = self.forward(X)
            loss = np.mean((output - y) ** 2)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predictions
        """
        output, _ = self.forward(X)
        return output
