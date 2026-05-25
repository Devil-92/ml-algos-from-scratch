"""
Logistic Regression for Binary Classification.

Mathematical Foundation:
- Sigmoid function: σ(z) = 1 / (1 + e^(-z))
- Log loss: L = -(y*log(ŷ) + (1-y)*log(1-ŷ))
- Gradient: dL/dw = X^T * (ŷ - y) / m
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')
from src.utils import MetricsMixin, sigmoid


class LogisticRegression(MetricsMixin):
    """
    Logistic Regression for binary classification.
    
    Uses sigmoid activation and log loss for binary classification.
    
    Attributes:
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations
        regularization: Type of regularization ('none', 'l1', 'l2')
        lambda_param: Regularization strength
        weights: Model coefficients
        bias: Model bias term
        loss_history: Loss at each iteration
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000,
                 regularization: str = 'none', lambda_param: float = 0.01):
        """
        Initialize Logistic Regression.
        
        Args:
            learning_rate: Learning rate for gradient descent
            iterations: Number of training iterations
            regularization: Type of regularization ('none', 'l1', 'l2')
            lambda_param: Regularization strength
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = 0
        self.loss_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Train the logistic regression model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) with values 0 or 1
            
        Returns:
            Self for method chaining
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.iterations):
            # Forward pass
            z = X.dot(self.weights) + self.bias
            y_pred = sigmoid(z)
            
            # Compute log loss
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -(1 / n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            
            # Add regularization
            if self.regularization == 'l2':
                loss += (self.lambda_param / (2 * n_samples)) * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                loss += (self.lambda_param / (2 * n_samples)) * np.sum(np.abs(self.weights))
            
            self.loss_history.append(loss)
            
            # Compute gradients
            error = y_pred - y
            dw = (1 / n_samples) * X.T.dot(error)
            db = (1 / n_samples) * np.sum(error)
            
            # Add regularization to gradient
            if self.regularization == 'l2':
                dw += (self.lambda_param / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_param / n_samples) * np.sign(self.weights)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions (values between 0 and 1).
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Probability of class 1 for each sample
        """
        if self.weights is None:
            raise ValueError("Model must be trained before prediction")
        
        z = X.dot(self.weights) + self.bias
        return sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Get class predictions (0 or 1).
        
        Args:
            X: Features (n_samples, n_features)
            threshold: Decision threshold for classification (default: 0.5)
            
        Returns:
            Predicted class labels (0 or 1)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy on test data.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict(X)
        return self.accuracy(y, y_pred)
