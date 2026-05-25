# Examples - ML Algorithms from Scratch

This directory contains working examples demonstrating all implemented ML algorithms.

## Running Examples

### Supervised Learning
```bash
python examples/supervised_learning_demo.py
```
Demonstrates:
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### Unsupervised Learning
```bash
python examples/unsupervised_learning_demo.py
```
Demonstrates:
- K-Means Clustering
- K-Means++ (improved initialization)
- Hierarchical Clustering (single, complete, average linkage)
- Principal Component Analysis (PCA)
- Isolation Forest (Anomaly Detection)

### Neural Networks
```bash
python examples/neural_networks_demo.py
```
Demonstrates:
- Multi-Layer Perceptron (MLP)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Convolutional Neural Network (CNN)

## Example Output

Each demo script outputs:
1. Algorithm initialization details
2. Training results and metrics
3. Sample predictions
4. Performance metrics (accuracy, R² score, inertia, etc.)

## Quick Start

```python
from src.supervised import LinearRegression
from src.utils import train_test_split

# Generate data
X = np.random.randn(100, 5)
y = np.random.randn(100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Evaluate
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

See individual demo scripts for more detailed examples!
