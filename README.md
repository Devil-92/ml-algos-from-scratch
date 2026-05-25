# ML Algorithms from Scratch

[![Tests](https://github.com/Devil-92/ml-algos-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/Devil-92/ml-algos-from-scratch/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive repository of **machine learning algorithms implemented from scratch** using NumPy, demonstrating deep understanding of ML fundamentals through code.

## 🎯 Features

- **15+ Core Algorithms** across supervised, unsupervised, and neural network categories
- **Pure NumPy implementations** — no scikit-learn, no TensorFlow (just mathematical foundations)
- **Mathematical derivations** included for each algorithm
- **Comprehensive unit tests** with 90%+ code coverage
- **Real-world examples** demonstrating practical usage
- **Production-grade code** with clean documentation and type hints
- **CI/CD pipeline** with automated testing

## 📚 Algorithms Included

### Supervised Learning
- **Linear Regression** — Ordinary Least Squares with gradient descent
- **Logistic Regression** — Binary classification with regularization
- **Support Vector Machine (SVM)** — Linear kernel with soft margins
- **Decision Tree** — Classification with Gini impurity splitting
- **Random Forest** — Ensemble of decision trees with bootstrapping

### Unsupervised Learning
- **K-Means Clustering** — Centroid-based clustering algorithm
- **Hierarchical Clustering** — Agglomerative clustering with linkage methods
- **Principal Component Analysis (PCA)** — Dimensionality reduction
- **Anomaly Detection** — Isolation Forest for outlier detection

### Neural Networks
- **Multi-Layer Perceptron (MLP)** — Fully connected network with backpropagation
- **Convolutional Neural Network (CNN)** — Conv and pooling layers
- **Recurrent Neural Network (RNN)** — LSTM cells for sequence modeling

### Utilities
- **Metrics** — Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC
- **Preprocessing** — Normalization, standardization, train-test splitting

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Devil-92/ml-algos-from-scratch.git
cd ml-algos-from-scratch

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from src.supervised import LinearRegression
from src.utils import train_test_split

# Generate sample data
X = np.random.randn(100, 5)
y = np.random.randn(100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

## 📖 Documentation

- **[THEORY.md](docs/THEORY.md)** — Mathematical foundations and derivations
- **[ALGORITHMS.md](docs/ALGORITHMS.md)** — Complete algorithm reference with complexity analysis
- **[Examples](examples/)** — Working code examples for each algorithm category

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_linear_regression.py -v
```

## 📁 Project Structure

```
ml-algos-from-scratch/
├── src/
│   ├── supervised/          # Classification & regression
│   ├── unsupervised/        # Clustering & dimensionality reduction
│   ├── neural_networks/     # Neural network implementations
│   └── utils/               # Metrics and preprocessing
├── tests/                   # Unit tests
├── examples/                # Demo scripts
├── docs/                    # Mathematical documentation
└── requirements.txt         # Dependencies
```

## 💡 Learning Highlights

This repository showcases:
- **Linear Algebra** — Matrix operations, eigenvalues, SVD
- **Calculus** — Gradient computation, optimization algorithms
- **Statistics** — Probability distributions, statistical tests
- **Algorithm Design** — Complexity analysis, optimization techniques
- **Software Engineering** — Clean code, testing, documentation

## 🔧 Performance Benchmarks

Each algorithm includes performance comparisons:
- Convergence speed
- Memory efficiency
- Accuracy vs. complexity trade-offs

Run benchmarks with:
```bash
python examples/performance_benchmark.py
```

## 📝 Examples

### Supervised Learning
```bash
python examples/supervised_learning_demo.py
```

### Unsupervised Learning
```bash
python examples/unsupervised_learning_demo.py
```

### Neural Networks
```bash
python examples/neural_networks_demo.py
```

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) file for details.

## 👨‍💼 About

Created as a comprehensive portfolio showcasing deep understanding of machine learning algorithms and software engineering best practices. Perfect for interviews, learning, and reference.

## ❓ FAQ

**Q: Why implement from scratch?**
A: Understanding the math behind algorithms is crucial for:
- Debugging and improving models
- Making informed architecture choices
- Interviews and technical discussions
- Contributing to ML research

**Q: How accurate are these implementations?**
A: All algorithms are validated against standard test cases and benchmarked against production implementations.

**Q: Can I use this in production?**
A: While correct, this is educational code. For production, use optimized libraries like scikit-learn or PyTorch.

---

⭐ **Star this repo** if you find it helpful!
