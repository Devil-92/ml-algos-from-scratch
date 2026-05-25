"""
K-Means Clustering Algorithm.

Mathematical Foundation:
- Objective: Minimize within-cluster sum of squares (WCSS)
- WCSS = Σ||x_i - c_k||² for each cluster k
- Algorithm:
  1. Initialize k centroids randomly
  2. Assign each point to nearest centroid
  3. Update centroids as mean of cluster points
  4. Repeat until convergence
"""

import numpy as np
from typing import Tuple
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')
from src.utils import euclidean_distance


class KMeans:
    """
    K-Means Clustering Algorithm.
    
    Attributes:
        k: Number of clusters
        max_iterations: Maximum iterations
        random_state: Random seed
        centroids: Cluster centroids
        labels: Cluster assignments
        inertia_: Within-cluster sum of squares
    """
    
    def __init__(self, k: int = 3, max_iterations: int = 100, random_state: int = None):
        """
        Initialize K-Means.
        
        Args:
            k: Number of clusters (default: 3)
            max_iterations: Max iterations (default: 100)
            random_state: Random seed for reproducibility
        """
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit K-Means clustering.
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly from data points
        random_indices = np.random.choice(n_samples, size=self.k, replace=False)
        self.centroids = X[random_indices].copy()
        
        for iteration in range(self.max_iterations):
            # Assign clusters
            self.labels = self._assign_clusters(X)
            
            # Store old centroids for convergence check
            old_centroids = self.centroids.copy()
            
            # Update centroids
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = cluster_points.mean(axis=0)
            
            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break
        
        # Calculate inertia
        self.inertia_ = np.sum([
            np.sum((X[self.labels == i] - self.centroids[i]) ** 2)
            for i in range(self.k)
        ])
        
        return self
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each point to nearest centroid.
        
        Args:
            X: Data points
            
        Returns:
            Cluster assignments
        """
        distances = np.zeros((X.shape[0], self.k))
        
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        
        return np.argmin(distances, axis=1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data points
            
        Returns:
            Cluster assignments
        """
        if self.centroids is None:
            raise ValueError("Model must be trained before prediction")
        
        return self._assign_clusters(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and return cluster labels.
        
        Args:
            X: Data points
            
        Returns:
            Cluster assignments
        """
        self.fit(X)
        return self.labels


class KMeansPlusPlus(KMeans):
    """K-Means with k-means++ initialization for better initial centroids."""
    
    def fit(self, X: np.ndarray) -> 'KMeansPlusPlus':
        """
        Fit K-Means with k-means++ initialization.
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        self.centroids = []
        
        # Choose first centroid randomly
        first_idx = np.random.choice(n_samples)
        self.centroids.append(X[first_idx])
        
        # Choose remaining k-1 centroids
        for _ in range(self.k - 1):
            # Compute distances to nearest centroid
            distances = np.array([
                min([np.linalg.norm(x - c) for c in self.centroids])
                for x in X
            ])
            
            # Probability proportional to distance squared
            probabilities = distances ** 2 / np.sum(distances ** 2)
            cumulative = np.cumsum(probabilities)
            r = np.random.rand()
            
            next_idx = np.searchsorted(cumulative, r)
            next_idx = min(next_idx, n_samples - 1)
            self.centroids.append(X[next_idx])
        
        self.centroids = np.array(self.centroids)
        
        # Continue with standard K-Means
        for iteration in range(self.max_iterations):
            self.labels = self._assign_clusters(X)
            old_centroids = self.centroids.copy()
            
            for i in range(self.k):
                cluster_points = X[self.labels == i]
                if len(cluster_points) > 0:
                    self.centroids[i] = cluster_points.mean(axis=0)
            
            if np.allclose(old_centroids, self.centroids):
                break
        
        self.inertia_ = np.sum([
            np.sum((X[self.labels == i] - self.centroids[i]) ** 2)
            for i in range(self.k)
        ])
        
        return self
