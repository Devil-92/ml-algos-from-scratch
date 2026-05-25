"""
Hierarchical Clustering and Anomaly Detection.

Hierarchical Clustering:
- Agglomerative approach: bottom-up
- Start with each point as cluster
- Iteratively merge closest clusters
- Linkage methods: single, complete, average

Anomaly Detection (Isolation Forest):
- Random trees that isolate anomalies
- Anomalies isolated faster than normal points
"""

import numpy as np
from typing import Optional, List, Tuple
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')


class HierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering.
    
    Attributes:
        n_clusters: Number of clusters to form
        linkage: Linkage method ('single', 'complete', 'average')
        labels_: Cluster assignments
        distances_: Distances between clusters
    """
    
    def __init__(self, n_clusters: int = 2, linkage: str = 'complete'):
        """
        Initialize Hierarchical Clustering.
        
        Args:
            n_clusters: Number of clusters (default: 2)
            linkage: Linkage method (default: 'complete')
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.distances_ = []
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and predict cluster labels.
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        n_samples = X.shape[0]
        
        # Initialize each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        
        while len(clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = np.inf
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(X, clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            self.distances_.append(min_dist)
            
            # Merge closest clusters
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
        
        # Assign labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_id in cluster:
                self.labels_[point_id] = cluster_id
        
        return self.labels_
    
    def _cluster_distance(self, X: np.ndarray, cluster1: List[int], 
                         cluster2: List[int]) -> float:
        """
        Calculate distance between two clusters.
        
        Args:
            X: Data points
            cluster1: Indices of first cluster
            cluster2: Indices of second cluster
            
        Returns:
            Distance between clusters
        """
        if self.linkage == 'single':
            # Minimum distance between any two points
            return min(
                np.linalg.norm(X[i] - X[j])
                for i in cluster1 for j in cluster2
            )
        elif self.linkage == 'complete':
            # Maximum distance between any two points
            return max(
                np.linalg.norm(X[i] - X[j])
                for i in cluster1 for j in cluster2
            )
        elif self.linkage == 'average':
            # Average distance between all pairs
            distances = [
                np.linalg.norm(X[i] - X[j])
                for i in cluster1 for j in cluster2
            ]
            return np.mean(distances)
        else:
            raise ValueError(f"Unknown linkage method: {self.linkage}")


class IsolationForest:
    """
    Isolation Forest for Anomaly Detection.
    
    Anomalies are isolated faster in random trees than normal points.
    
    Attributes:
        n_trees: Number of isolation trees
        sample_size: Size of sample for each tree
        max_depth: Maximum tree depth
        random_state: Random seed
        trees_: List of isolation trees
        scores_: Anomaly scores
    """
    
    def __init__(self, n_trees: int = 100, sample_size: Optional[int] = None,
                 max_depth: int = 20, random_state: int = None):
        """
        Initialize Isolation Forest.
        
        Args:
            n_trees: Number of trees (default: 100)
            sample_size: Sample size for each tree (default: min(256, n_samples))
            max_depth: Maximum tree depth (default: 20)
            random_state: Random seed
        """
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = []
        self.scores_ = None
    
    def fit(self, X: np.ndarray) -> 'IsolationForest':
        """
        Train isolation forest.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.sample_size is None:
            self.sample_size = min(256, X.shape[0])
        
        n_samples = X.shape[0]
        
        for _ in range(self.n_trees):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=self.sample_size, replace=False)
            X_sample = X[indices]
            
            # Build isolation tree
            tree = self._build_tree(X_sample, depth=0)
            self.trees_.append(tree)
        
        return self
    
    def _build_tree(self, X: np.ndarray, depth: int) -> dict:
        """
        Build isolation tree recursively.
        
        Args:
            X: Data points
            depth: Current depth
            
        Returns:
            Tree node (dict)
        """
        if depth >= self.max_depth or X.shape[0] <= 1:
            return {'type': 'leaf', 'size': X.shape[0]}
        
        # Randomly select feature and split value
        feature = np.random.choice(X.shape[1])
        split_value = np.random.uniform(X[:, feature].min(), X[:, feature].max())
        
        # Split data
        left_mask = X[:, feature] <= split_value
        X_left = X[left_mask]
        X_right = X[~left_mask]
        
        # Handle case where no split occurs
        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            return {'type': 'leaf', 'size': X.shape[0]}
        
        return {
            'type': 'internal',
            'feature': feature,
            'split': split_value,
            'left': self._build_tree(X_left, depth + 1),
            'right': self._build_tree(X_right, depth + 1)
        }
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Args:
            X: Data points
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        scores = np.zeros(X.shape[0])
        
        for x_idx, x in enumerate(X):
            path_lengths = []
            for tree in self.trees_:
                path_lengths.append(self._path_length(x, tree, 0))
            
            # Average path length
            avg_path_length = np.mean(path_lengths)
            # Normalize by average path length for random data
            c = np.mean([self._avg_path_length(X.shape[0]) for _ in range(1)])
            scores[x_idx] = 2 ** (-avg_path_length / c)
        
        return scores
    
    def _path_length(self, x: np.ndarray, tree: dict, depth: int) -> float:
        """
        Calculate path length from root to leaf.
        
        Args:
            x: Data point
            tree: Tree node
            depth: Current depth
            
        Returns:
            Path length
        """
        if tree['type'] == 'leaf':
            return depth + self._c(tree['size'])
        
        if x[tree['feature']] <= tree['split']:
            return self._path_length(x, tree['left'], depth + 1)
        else:
            return self._path_length(x, tree['right'], depth + 1)
    
    @staticmethod
    def _c(n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    
    @staticmethod
    def _avg_path_length(n: int) -> float:
        """Expected average path length."""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Data points
            threshold: Anomaly score threshold
            
        Returns:
            Labels (1 = anomaly, -1 = normal)
        """
        scores = self.decision_function(X)
        return np.where(scores > threshold, 1, -1)
