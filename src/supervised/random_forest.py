"""
Random Forest Classifier.

Ensemble method combining multiple decision trees:
- Bootstrap sampling: Sample with replacement from training data
- Feature subsampling: Each split considers random subset of features
- Majority voting: Aggregate predictions from all trees
"""

import numpy as np
from typing import List
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')
from src.supervised.decision_tree import DecisionTreeClassifier
from src.utils import MetricsMixin


class RandomForestClassifier(MetricsMixin):
    """
    Random Forest Classifier using bootstrap aggregating.
    
    Attributes:
        n_trees: Number of trees in forest
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        trees: List of trained decision trees
    """
    
    def __init__(self, n_trees: int = 10, max_depth: int = 10, 
                 min_samples_split: int = 2, random_state: int = None):
        """
        Initialize Random Forest.
        
        Args:
            n_trees: Number of trees (default: 10)
            max_depth: Maximum tree depth (default: 10)
            min_samples_split: Minimum samples to split (default: 2)
            random_state: Random seed for reproducibility
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """
        Train random forest.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            Self for method chaining
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Train tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                        min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using majority voting.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        if not self.trees:
            raise ValueError("Model must be trained before prediction")
        
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Majority voting
        return np.array([np.bincount(predictions[:, i]).argmax() 
                        for i in range(predictions.shape[1])])
    
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
    
    def feature_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Estimate feature importance using permutation importance.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Importance score for each feature
        """
        baseline_score = self.score(X, y)
        importances = np.zeros(X.shape[1])
        
        for feature in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature])
            permuted_score = self.score(X_permuted, y)
            importances[feature] = baseline_score - permuted_score
        
        return importances / np.sum(importances)
