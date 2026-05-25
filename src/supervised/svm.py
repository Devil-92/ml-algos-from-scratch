"""
Support Vector Machine (SVM) for Binary Classification.

Mathematical Foundation:
- Objective: Maximize margin between classes
- Constraint: y_i(w·x_i + b) ≥ 1 - ξ_i (soft margin)
- Hinge loss: L = 1/m * Σmax(0, 1 - y_i*f(x_i)) + λ||w||²
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')
from src.utils import MetricsMixin


class SVM(MetricsMixin):
    """
    Support Vector Machine for binary classification.
    
    Uses hinge loss and stochastic gradient descent for training.
    
    Attributes:
        learning_rate: Learning rate for SGD
        iterations: Number of training iterations
        lambda_param: Regularization strength
        weights: Model weights
        bias: Bias term
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000,
                 lambda_param: float = 0.01):
        """
        Initialize SVM.
        
        Args:
            learning_rate: Learning rate for SGD
            iterations: Number of iterations
            lambda_param: Regularization parameter
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = 0
        self.loss_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Train SVM using stochastic gradient descent.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) with values -1 or 1
            
        Returns:
            Self for method chaining
        """
        # Convert labels to -1, 1 if needed
        y = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.iterations):
            loss = 0
            for i in range(n_samples):
                # Forward pass
                z = np.dot(self.weights, X[i]) + self.bias
                
                # Hinge loss
                if y[i] * z < 1:
                    # Misclassified or within margin
                    self.weights -= self.learning_rate * (self.lambda_param * self.weights - y[i] * X[i])
                    self.bias -= self.learning_rate * (-y[i])
                    loss += 1 - y[i] * z
                else:
                    # Correctly classified
                    self.weights -= self.learning_rate * self.lambda_param * self.weights
                
                # Regularization term for loss calculation
                loss += self.lambda_param * np.sum(self.weights ** 2)
            
            self.loss_history.append(loss / n_samples)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted labels (-1 or 1)
        """
        if self.weights is None:
            raise ValueError("Model must be trained before prediction")
        
        z = X.dot(self.weights) + self.bias
        return np.where(z < 0, -1, 1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision function values.
        
        Args:
            X: Features
            
        Returns:
            Decision function values
        """
        return X.dot(self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy.
        
        Args:
            X: Test features
            y: Test labels (0 or 1)
            
        Returns:
            Accuracy score
        """
        # Convert labels to -1, 1 if needed
        y = np.where(y <= 0, -1, 1)
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
