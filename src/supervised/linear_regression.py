"""
Linear Regression using Gradient Descent.

Mathematical Foundation:
- Objective: Minimize Mean Squared Error (MSE)
- Loss function: L = (1/2m) * Σ(y_pred - y_true)²
- Gradient: dL/dw = (1/m) * X^T * (y_pred - y_true)
- Update rule: w := w - α * dL/dw
"""

import numpy as np
from typing import Optional
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')
from src.utils import MetricsMixin


class LinearRegression(MetricsMixin):
    """
    Linear Regression using Gradient Descent.
    
    Fits a linear model: y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
    
    Attributes:
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations
        regularization: Type of regularization ('none', 'l1', 'l2')
        lambda_param: Regularization strength
        weights: Model coefficients (w)
        bias: Model bias term (b/w₀)
        loss_history: Loss at each iteration
    """
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000,
                 regularization: str = 'none', lambda_param: float = 0.01):
        """
        Initialize Linear Regression.
        
        Args:
            learning_rate: Learning rate for gradient descent (default: 0.01)
            iterations: Number of training iterations (default: 1000)
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
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Train the linear regression model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training target (n_samples,)
            
        Returns:
            Self for method chaining
        """
        n_samples, n_features = X.shape
        
        # Initialize weights to small random values
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent loop
        for iteration in range(self.iterations):
            # Forward pass
            y_pred = X.dot(self.weights) + self.bias
            
            # Compute loss
            mse = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            
            # Add regularization term to loss
            if self.regularization == 'l2':
                mse += (self.lambda_param / (2 * n_samples)) * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                mse += (self.lambda_param / (2 * n_samples)) * np.sum(np.abs(self.weights))
            
            self.loss_history.append(mse)
            
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model must be trained before prediction")
        
        return X.dot(self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score on test data.
        
        Args:
            X: Test features
            y: Test target
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        return self.r_squared(y, y_pred)


class LinearRegressionClosedForm:
    """
    Linear Regression using Closed-Form Solution (Normal Equation).
    
    w = (X^T X)^(-1) X^T y
    
    More efficient for small datasets, but O(n³) complexity for matrix inversion.
    """
    
    def __init__(self):
        """Initialize Linear Regression with closed-form solution."""
        self.weights = None
        self.bias = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionClosedForm':
        """
        Train using normal equation.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training target (n_samples,)
            
        Returns:
            Self for method chaining
        """
        # Add bias term (column of 1s)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: w = (X^T X)^(-1) X^T y
        try:
            w = np.linalg.inv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            w = np.linalg.pinv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(y)
        
        self.bias = w[0]
        self.weights = w[1:]
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model must be trained before prediction")
        
        return X.dot(self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.
        
        Args:
            X: Test features
            y: Test target
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
