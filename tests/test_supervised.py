"""
Tests for supervised learning algorithms.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')
from src.supervised import (LinearRegression, LogisticRegression, 
                            DecisionTreeClassifier, RandomForestClassifier, SVM)
from src.utils import train_test_split


class TestLinearRegression:
    """Tests for Linear Regression."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic regression data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
        return X, y
    
    def test_fit_predict(self, data):
        """Test model fitting and prediction."""
        X, y = data
        model = LinearRegression(learning_rate=0.1, iterations=100)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        assert y_pred.shape == y.shape
        assert not np.any(np.isnan(y_pred))
    
    def test_score(self, data):
        """Test R² score calculation."""
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = LinearRegression(learning_rate=0.1, iterations=1000)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        assert 0 <= score <= 1


class TestLogisticRegression:
    """Tests for Logistic Regression."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    def test_fit_predict(self, data):
        """Test model fitting and prediction."""
        X, y = data
        model = LogisticRegression(learning_rate=0.1, iterations=100)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        assert y_pred.shape == y.shape
        assert np.all((y_pred == 0) | (y_pred == 1))
    
    def test_predict_proba(self, data):
        """Test probability predictions."""
        X, y = data
        model = LogisticRegression(iterations=100)
        model.fit(X, y)
        proba = model.predict_proba(X)
        
        assert proba.shape == (X.shape[0],)
        assert np.all((proba >= 0) & (proba <= 1))
    
    def test_score(self, data):
        """Test accuracy score."""
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model = LogisticRegression(learning_rate=0.1, iterations=500)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        assert 0 <= score <= 1


class TestDecisionTree:
    """Tests for Decision Tree Classifier."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        return X, y
    
    def test_fit_predict(self, data):
        """Test tree fitting and prediction."""
        X, y = data
        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        assert y_pred.shape == y.shape
        assert np.all((y_pred == 0) | (y_pred == 1))
    
    def test_score(self, data):
        """Test accuracy score."""
        X, y = data
        model = DecisionTreeClassifier(max_depth=5)
        model.fit(X, y)
        score = model.score(X, y)
        
        assert 0 <= score <= 1
    
    def test_tree_depth(self, data):
        """Test max_depth constraint."""
        X, y = data
        model = DecisionTreeClassifier(max_depth=2)
        model.fit(X, y)
        
        def get_depth(node):
            if node.is_leaf():
                return 0
            return 1 + max(get_depth(node.left), get_depth(node.right))
        
        depth = get_depth(model.root)
        assert depth <= 2


class TestRandomForest:
    """Tests for Random Forest Classifier."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    def test_fit_predict(self, data):
        """Test forest fitting and prediction."""
        X, y = data
        model = RandomForestClassifier(n_trees=5, max_depth=5)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        assert y_pred.shape == y.shape
        assert np.all((y_pred == 0) | (y_pred == 1))
    
    def test_n_trees(self, data):
        """Test number of trees."""
        X, y = data
        model = RandomForestClassifier(n_trees=10, max_depth=5)
        model.fit(X, y)
        
        assert len(model.trees) == 10
    
    def test_feature_importance(self, data):
        """Test feature importance calculation."""
        X, y = data
        model = RandomForestClassifier(n_trees=5, max_depth=5)
        model.fit(X, y)
        
        importances = model.feature_importance(X, y)
        assert importances.shape == (X.shape[1],)
        assert np.isclose(np.sum(importances), 1.0)


class TestSVM:
    """Tests for Support Vector Machine."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        return X, y
    
    def test_fit_predict(self, data):
        """Test SVM fitting and prediction."""
        X, y = data
        model = SVM(learning_rate=0.01, iterations=100)
        model.fit(X, y)
        y_pred = model.predict(X)
        
        assert y_pred.shape == y.shape
        assert np.all((y_pred == -1) | (y_pred == 1))
    
    def test_decision_function(self, data):
        """Test decision function."""
        X, y = data
        model = SVM(learning_rate=0.01, iterations=100)
        model.fit(X, y)
        
        z = model.decision_function(X)
        assert z.shape == (X.shape[0],)
    
    def test_score(self, data):
        """Test accuracy score."""
        X, y = data
        model = SVM(learning_rate=0.01, iterations=100)
        model.fit(X, y)
        score = model.score(X, y)
        
        assert 0 <= score <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
