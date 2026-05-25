"""
Neural Network Examples - Demonstrating neural network implementations.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')

from src.neural_networks import MLP, one_hot_encode, LSTM, SimpleRNN, SimpleCNN


def main():
    """Run neural network examples."""
    
    print("=" * 70)
    print("ML ALGORITHMS FROM SCRATCH - NEURAL NETWORK EXAMPLES")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Classification dataset
    print("\n1. MULTI-LAYER PERCEPTRON (MLP)")
    print("-" * 70)
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 3, 100)
    y_train_encoded = one_hot_encode(y_train, 3)
    
    X_test = np.random.randn(20, 20)
    y_test = np.random.randint(0, 3, 20)
    
    model = MLP(layer_sizes=[20, 32, 16, 3], learning_rate=0.1, iterations=100)
    model.fit(X_train, y_train_encoded)
    
    accuracy = model.score(X_test, y_test)
    print(f"MLP Architecture: [20 -> 32 -> 16 -> 3]")
    print(f"Final Training Loss: {model.loss_history[-1]:.4f}")
    print(f"Initial Training Loss: {model.loss_history[0]:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Get probability predictions
    proba = model.predict_proba(X_test[:3])
    print(f"Sample probability predictions:\n{proba.round(3)}")
    
    print("\n2. RECURRENT NEURAL NETWORK (RNN)")
    print("-" * 70)
    X_seq = np.random.randn(10, 5, 8)  # (seq_len, batch_size, features)
    y_seq = np.random.randn(5, 3)
    
    rnn = SimpleRNN(input_dim=8, hidden_dim=16, output_dim=3, iterations=10)
    rnn.fit(X_seq, y_seq)
    
    predictions = rnn.predict(X_seq)
    print(f"RNN Input shape: {X_seq.shape} (sequence_length, batch_size, features)")
    print(f"RNN Output shape: {predictions.shape}")
    print(f"Sample predictions:\n{predictions[:3].round(3)}")
    
    print("\n3. LSTM (LONG SHORT-TERM MEMORY)")
    print("-" * 70)
    lstm = LSTM(input_dim=8, hidden_dim=16)
    
    h, c = lstm.forward(X_seq)
    print(f"LSTM Input shape: {X_seq.shape}")
    print(f"LSTM Hidden state shape: {h.shape}")
    print(f"LSTM Cell state shape: {c.shape}")
    print("LSTM successfully processes sequential data!")
    
    print("\n4. CONVOLUTIONAL NEURAL NETWORK (CNN)")
    print("-" * 70)
    X_images = np.random.randn(4, 28, 28, 1)  # (batch, height, width, channels)
    
    cnn = SimpleCNN(output_dim=10)
    features = cnn.predict(X_images)
    
    print(f"CNN Input shape (images): {X_images.shape}")
    print(f"CNN Output shape (features): {features.shape}")
    print("CNN successfully extracts features from images!")
    
    print("\n" + "=" * 70)
    print("All neural network architectures demonstrated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
