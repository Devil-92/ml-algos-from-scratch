"""
Multi-Layer Perceptron (Fully Connected Neural Network).

Mathematical Foundation:
- Forward pass: a = σ(Wx + b) for each layer
- Backpropagation: ∂L/∂w = (∂L/∂a)(∂a/∂z)(∂z/∂w)
- Gradient descent: w := w - α(∂L/∂w)
"""

import numpy as np
from typing import List, Tuple, Callable
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')
from src.utils import relu, relu_derivative, sigmoid, softmax


class MLP:
    """
    Multi-Layer Perceptron using backpropagation.
    
    Attributes:
        layer_sizes: Sizes of each layer
        learning_rate: Learning rate for SGD
        iterations: Number of training iterations
        weights: Layer weights
        biases: Layer biases
        loss_history: Loss at each epoch
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01,
                 iterations: int = 1000, activation: str = 'relu'):
        """
        Initialize MLP.
        
        Args:
            layer_sizes: List of layer sizes (e.g., [784, 128, 64, 10])
            learning_rate: Learning rate (default: 0.01)
            iterations: Number of iterations (default: 1000)
            activation: Activation function ('relu' or 'sigmoid')
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.activation = activation
        self.weights = []
        self.biases = []
        self.loss_history = []
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize weights and biases with He initialization."""
        np.random.seed(42)
        
        for i in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]
            
            # He initialization
            w = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
            b = np.zeros((1, out_dim))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass.
        
        Args:
            X: Input data
            
        Returns:
            Activations and pre-activations for each layer
        """
        activations = [X]
        pre_activations = []
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = activations[-1].dot(self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            
            if self.activation == 'relu':
                a = relu(z)
            else:
                a = sigmoid(z)
            
            activations.append(a)
        
        # Output layer (softmax)
        z = activations[-1].dot(self.weights[-1]) + self.biases[-1]
        pre_activations.append(z)
        a = softmax(z)
        activations.append(a)
        
        return activations, pre_activations
    
    def _backward(self, X: np.ndarray, y: np.ndarray, 
                  activations: List[np.ndarray], 
                  pre_activations: List[np.ndarray]):
        """
        Backward pass (backpropagation).
        
        Args:
            X: Input data
            y: Target labels (one-hot encoded)
            activations: Layer activations from forward pass
            pre_activations: Pre-activations from forward pass
        """
        m = X.shape[0]
        
        # Output layer error
        dz = activations[-1] - y
        
        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights and biases
            dw = activations[i].T.dot(dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            
            # Update parameters
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
            
            if i > 0:
                # Propagate error to previous layer
                dz = dz.dot(self.weights[i].T)
                
                if self.activation == 'relu':
                    dz *= relu_derivative(pre_activations[i - 1])
                else:
                    dz *= (activations[i] * (1 - activations[i]))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLP':
        """
        Train MLP.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples, n_classes) one-hot encoded
            
        Returns:
            Self for method chaining
        """
        for epoch in range(self.iterations):
            # Forward pass
            activations, pre_activations = self._forward(X)
            
            # Compute loss
            predictions = activations[-1]
            loss = -np.mean(y * np.log(predictions + 1e-8))
            self.loss_history.append(loss)
            
            # Backward pass
            self._backward(X, y, activations, pre_activations)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predicted class labels
        """
        activations, _ = self._forward(X)
        predictions = activations[-1]
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Features
            
        Returns:
            Class probabilities
        """
        activations, _ = self._forward(X)
        return activations[-1]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy.
        
        Args:
            X: Test features
            y: Test labels (0-indexed class labels)
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    One-hot encode labels.
    
    Args:
        y: Class labels (n_samples,)
        n_classes: Number of classes
        
    Returns:
        One-hot encoded labels (n_samples, n_classes)
    """
    one_hot = np.zeros((y.shape[0], n_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot
