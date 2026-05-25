"""
Principal Component Analysis (PCA) for Dimensionality Reduction.

Mathematical Foundation:
- Objective: Find directions of maximum variance
- Steps:
  1. Center data: X_centered = X - mean(X)
  2. Compute covariance matrix: Cov = X_centered^T * X_centered / n
  3. Find eigenvalues and eigenvectors of covariance matrix
  4. Project data onto top k eigenvectors
"""

import numpy as np
from typing import Optional
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')


class PCA:
    """
    Principal Component Analysis for dimensionality reduction.
    
    Attributes:
        n_components: Number of principal components
        components_: Principal components (eigenvectors)
        explained_variance_: Variance explained by each component
        mean_: Mean of training data
    """
    
    def __init__(self, n_components: int = 2):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of components to keep (default: 2)
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA model.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top n_components
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto principal components.
        
        Args:
            X: Data to transform (n_samples, n_features)
            
        Returns:
            Transformed data (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("Model must be trained before transform")
        
        X_centered = X - self.mean_
        return X_centered.dot(self.components_)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and transform data.
        
        Args:
            X: Training data
            
        Returns:
            Transformed data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Reconstruct original data from transformed data.
        
        Args:
            X_transformed: Transformed data
            
        Returns:
            Reconstructed data
        """
        return X_transformed.dot(self.components_.T) + self.mean_
    
    def explained_variance_ratio(self) -> np.ndarray:
        """
        Get fraction of variance explained by each component.
        
        Returns:
            Explained variance ratio for each component
        """
        return self.explained_variance_ / np.sum(self.explained_variance_)
    
    def cumulative_explained_variance(self) -> np.ndarray:
        """
        Get cumulative explained variance.
        
        Returns:
            Cumulative explained variance ratio
        """
        return np.cumsum(self.explained_variance_ratio())
