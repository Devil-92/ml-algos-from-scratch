"""
Tests for unsupervised learning algorithms.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')
from src.unsupervised import (KMeans, KMeansPlusPlus, PCA, 
                              HierarchicalClustering, IsolationForest)


class TestKMeans:
    """Tests for K-Means clustering."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic clustering data."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [5, 5],
            np.random.randn(30, 2) + [10, 0]
        ])
        return X
    
    def test_fit_predict(self, data):
        """Test K-Means fitting and prediction."""
        model = KMeans(k=3, random_state=42)
        labels = model.fit_predict(data)
        
        assert labels.shape == (data.shape[0],)
        assert len(np.unique(labels)) == 3
    
    def test_centroids_shape(self, data):
        """Test centroids shape."""
        model = KMeans(k=3)
        model.fit(data)
        
        assert model.centroids.shape == (3, 2)
    
    def test_predict(self, data):
        """Test prediction on new data."""
        model = KMeans(k=3)
        model.fit(data)
        
        new_data = np.array([[0, 0], [5, 5], [10, 0]])
        predictions = model.predict(new_data)
        
        assert predictions.shape == (3,)


class TestKMeansPlusPlus:
    """Tests for K-Means++."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic data."""
        np.random.seed(42)
        return np.random.randn(100, 2)
    
    def test_initialization(self, data):
        """Test k-means++ initialization."""
        model = KMeansPlusPlus(k=3, random_state=42)
        model.fit(data)
        
        assert model.centroids.shape == (3, 2)
        assert model.inertia_ is not None


class TestPCA:
    """Tests for PCA."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        return X
    
    def test_fit_transform(self, data):
        """Test PCA fitting and transformation."""
        model = PCA(n_components=2)
        transformed = model.fit_transform(data)
        
        assert transformed.shape == (data.shape[0], 2)
    
    def test_explained_variance(self, data):
        """Test explained variance calculation."""
        model = PCA(n_components=3)
        model.fit(data)
        
        ratio = model.explained_variance_ratio()
        assert len(ratio) == 3
        assert np.all(ratio >= 0)
        assert np.isclose(np.sum(ratio), 1.0)
    
    def test_inverse_transform(self, data):
        """Test inverse transformation."""
        model = PCA(n_components=5)
        transformed = model.fit_transform(data)
        reconstructed = model.inverse_transform(transformed)
        
        assert reconstructed.shape == data.shape
        # Should be close but not exact
        mse = np.mean((data - reconstructed) ** 2)
        assert mse > 0


class TestHierarchicalClustering:
    """Tests for Hierarchical Clustering."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic data."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(20, 2) + [0, 0],
            np.random.randn(20, 2) + [5, 5]
        ])
        return X
    
    def test_fit_predict_single(self, data):
        """Test with single linkage."""
        model = HierarchicalClustering(n_clusters=2, linkage='single')
        labels = model.fit_predict(data)
        
        assert labels.shape == (data.shape[0],)
        assert len(np.unique(labels)) == 2
    
    def test_fit_predict_complete(self, data):
        """Test with complete linkage."""
        model = HierarchicalClustering(n_clusters=2, linkage='complete')
        labels = model.fit_predict(data)
        
        assert labels.shape == (data.shape[0],)
        assert len(np.unique(labels)) == 2
    
    def test_fit_predict_average(self, data):
        """Test with average linkage."""
        model = HierarchicalClustering(n_clusters=2, linkage='average')
        labels = model.fit_predict(data)
        
        assert labels.shape == (data.shape[0],)


class TestIsolationForest:
    """Tests for Isolation Forest."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic data with anomalies."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Add anomalies
        X[-5:] = np.random.uniform(-10, 10, (5, 2))
        return X
    
    def test_fit(self, data):
        """Test Isolation Forest fitting."""
        model = IsolationForest(n_trees=10, random_state=42)
        model.fit(data)
        
        assert len(model.trees_) == 10
    
    def test_decision_function(self, data):
        """Test anomaly score calculation."""
        model = IsolationForest(n_trees=10, random_state=42)
        model.fit(data)
        
        scores = model.decision_function(data)
        assert scores.shape == (data.shape[0],)
        assert np.all((scores >= 0) & (scores <= 1))
    
    def test_predict(self, data):
        """Test anomaly prediction."""
        model = IsolationForest(n_trees=10, random_state=42)
        model.fit(data)
        
        predictions = model.predict(data)
        assert predictions.shape == (data.shape[0],)
        assert np.all((predictions == -1) | (predictions == 1))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
