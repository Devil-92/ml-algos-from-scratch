"""
Utility functions for ML algorithms.
Includes metrics, preprocessing, and data manipulation utilities.
"""

import numpy as np
from typing import Tuple


class MetricsMixin:
    """Common metrics for evaluation."""
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score between 0 and 1
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix for binary classification.
        
        Args:
            y_true: Ground truth binary labels (0 or 1)
            y_pred: Predicted binary labels (0 or 1)
            
        Returns:
            2x2 confusion matrix [[TN, FP], [FN, TP]]
        """
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        return np.array([[tn, fp], [fn, tp]])
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate precision score: TP / (TP + FP)
        
        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Precision score between 0 and 1
        """
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate recall score: TP / (TP + FN)
        
        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted binary labels
            
        Returns:
            Recall score between 0 and 1
        """
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        
        Args:
            y_true: Ground truth binary labels
            y_pred: Predicted binary labels
            
        Returns:
            F1 score between 0 and 1
        """
        precision = MetricsMixin.precision(y_true, y_pred)
        recall = MetricsMixin.recall(y_true, y_pred)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate mean squared error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            MSE value
        """
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate root mean squared error.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² (coefficient of determination).
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            R² score between -∞ and 1
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - (ss_res / ss_tot)


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                    random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into training and testing sets.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        test_size: Proportion of dataset to include in test split (0-1)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    test_size_n = int(n_samples * test_size)
    
    # Random permutation of indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_size_n]
    train_indices = indices[test_size_n:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def normalize(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalize data to [0, 1] range.
    
    Args:
        X: Input data
        axis: Axis along which to normalize
        
    Returns:
        Normalized data
    """
    X_min = np.min(X, axis=axis, keepdims=True)
    X_max = np.max(X, axis=axis, keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)


def standardize(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Standardize data (zero mean, unit variance).
    
    Args:
        X: Input data
        axis: Axis along which to standardize
        
    Returns:
        Standardized data
    """
    X_mean = np.mean(X, axis=axis, keepdims=True)
    X_std = np.std(X, axis=axis, keepdims=True)
    return (X - X_mean) / (X_std + 1e-8)


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        x1: First point
        x2: Second point
        
    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        x1: First vector
        x2: Second vector
        
    Returns:
        Cosine similarity between -1 and 1
    """
    dot_product = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    return dot_product / (norm1 * norm2 + 1e-8)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid of input
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU activation function.
    
    Args:
        x: Input array
        
    Returns:
        ReLU of input
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU.
    
    Args:
        x: Input array
        
    Returns:
        Derivative of ReLU
    """
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function.
    
    Args:
        x: Input array (can be 1D or 2D)
        
    Returns:
        Softmax of input
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
