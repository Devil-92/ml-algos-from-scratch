"""
Decision Tree Classifier.

Mathematical Foundation:
- Gini impurity: G = 1 - Σ(p_i)² where p_i is proportion of class i
- Information gain: IG = G(parent) - Σ(n_child/n_parent * G(child))
- Recursively split on feature that maximizes information gain
"""

import numpy as np
from typing import Optional, List, Tuple
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')
from src.utils import MetricsMixin


class Node:
    """Node in decision tree."""
    
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None,
                 value: Optional[int] = None):
        """
        Initialize tree node.
        
        Args:
            feature: Feature index for split
            threshold: Threshold value for split
            left: Left child node
            right: Right child node
            value: Class label if leaf node
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return self.value is not None


class DecisionTreeClassifier(MetricsMixin):
    """
    Decision Tree Classifier using Gini impurity.
    
    Attributes:
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples required to split a node
        root: Root node of the tree
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        """
        Initialize Decision Tree Classifier.
        
        Args:
            max_depth: Maximum tree depth (default: 10)
            min_samples_split: Minimum samples to split (default: 2)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Build decision tree.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        self.root = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively build the decision tree.
        
        Args:
            X: Features
            y: Labels
            depth: Current tree depth
            
        Returns:
            Tree node
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        # Try all features
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Split
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                # Calculate gain
                parent_gini = self._gini(y)
                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                
                gain = parent_gini - (left_mask.sum() / n_samples * left_gini +
                                      right_mask.sum() / n_samples * right_gini)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        # No good split found
        if best_feature is None:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)
        
        # Recursively build left and right subtrees
        left_mask = X[:, best_feature] <= best_threshold
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)
    
    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """
        Calculate Gini impurity.
        
        Args:
            y: Labels
            
        Returns:
            Gini impurity value
        """
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        if self.root is None:
            raise ValueError("Model must be trained before prediction")
        
        return np.array([self._traverse(x, self.root) for x in X])
    
    def _traverse(self, x: np.ndarray, node: Node) -> int:
        """
        Traverse tree to get prediction for a sample.
        
        Args:
            x: Sample features
            node: Current node
            
        Returns:
            Predicted class
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
    
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
