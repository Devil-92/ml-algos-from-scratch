"""
Tests for neural network implementations.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')
from src.neural_networks import (MLP, one_hot_encode, LSTM, SimpleRNN, SimpleCNN,
                                  ConvLayer, PoolingLayer)


class TestMLP:
    """Tests for Multi-Layer Perceptron."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic classification data."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        return X, y
    
    def test_initialization(self):
        """Test MLP initialization."""
        model = MLP([10, 32, 16, 3])
        
        assert len(model.weights) == 3
        assert len(model.biases) == 3
    
    def test_forward_pass(self, data):
        """Test forward pass."""
        X, _ = data
        model = MLP([10, 16, 3])
        
        activations, pre_activations = model._forward(X)
        
        assert len(activations) == 3  # Input + 2 layers
        assert activations[-1].shape == (X.shape[0], 3)
    
    def test_one_hot_encode(self, data):
        """Test one-hot encoding."""
        _, y = data
        encoded = one_hot_encode(y, 3)
        
        assert encoded.shape == (len(y), 3)
        assert np.all((encoded == 0) | (encoded == 1))
        assert np.allclose(np.sum(encoded, axis=1), 1)
    
    def test_fit_predict(self, data):
        """Test MLP fitting and prediction."""
        X, y = data
        y_encoded = one_hot_encode(y, 3)
        
        model = MLP([10, 16, 8, 3], iterations=50)
        model.fit(X, y_encoded)
        predictions = model.predict(X)
        
        assert predictions.shape == (X.shape[0],)
        assert np.all((predictions >= 0) & (predictions < 3))
    
    def test_loss_decreases(self, data):
        """Test that loss decreases during training."""
        X, y = data
        y_encoded = one_hot_encode(y, 3)
        
        model = MLP([10, 16, 3], iterations=100)
        model.fit(X, y_encoded)
        
        # Loss should generally decrease
        assert model.loss_history[-1] < model.loss_history[0]


class TestRNN:
    """Tests for RNN."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic sequence data."""
        np.random.seed(42)
        X = np.random.randn(10, 5, 8)  # (seq_len, batch, features)
        y = np.random.randn(5, 3)      # (batch, output)
        return X, y
    
    def test_lstm_initialization(self):
        """Test LSTM initialization."""
        lstm = LSTM(input_dim=8, hidden_dim=16)
        
        assert lstm.Wf.shape == (8 + 16, 16)
        assert lstm.bf.shape == (1, 16)
    
    def test_lstm_forward(self, data):
        """Test LSTM forward pass."""
        X, _ = data
        lstm = LSTM(input_dim=8, hidden_dim=16)
        
        h, c = lstm.forward(X)
        
        assert h.shape == (X.shape[1], 16)
        assert c.shape == (X.shape[1], 16)
    
    def test_rnn_forward(self, data):
        """Test SimpleRNN forward pass."""
        X, y = data
        rnn = SimpleRNN(input_dim=8, hidden_dim=16, output_dim=3)
        
        output, hiddens = rnn.forward(X)
        
        assert output.shape == y.shape
        assert len(hiddens) == X.shape[0]


class TestCNN:
    """Tests for CNN."""
    
    @pytest.fixture
    def data(self):
        """Generate synthetic image data."""
        np.random.seed(42)
        X = np.random.randn(4, 28, 28, 1)  # (batch, height, width, channels)
        return X
    
    def test_conv_layer(self, data):
        """Test convolutional layer."""
        X = data
        conv = ConvLayer(n_filters=32, kernel_size=3, stride=1, padding=0)
        
        output = conv.forward(X)
        
        assert output.shape[3] == 32  # Number of filters
        assert output.shape[0] == 4   # Batch size
    
    def test_pooling_layer(self, data):
        """Test pooling layer."""
        X = data
        pool = PoolingLayer(pool_size=2, stride=2)
        
        output = pool.forward(X)
        
        assert output.shape[1] == 14  # Height halved
        assert output.shape[2] == 14  # Width halved
    
    def test_simple_cnn(self, data):
        """Test SimpleCNN."""
        X = data
        cnn = SimpleCNN(output_dim=10)
        
        features = cnn.predict(X)
        
        assert features.shape[0] == 4  # Batch size
        assert features.ndim == 2      # Flattened features


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
