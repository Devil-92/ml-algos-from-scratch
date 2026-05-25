# Algorithm Reference Guide

Complete reference for all implemented algorithms with usage examples and complexity analysis.

## Supervised Learning

### Linear Regression

**Purpose**: Predict continuous values

**Usage**:
```python
from src.supervised import LinearRegression

model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
r2 = model.score(X_test, y_test)
```

**Parameters**:
- `learning_rate`: Step size for gradient descent (default: 0.01)
- `iterations`: Number of training iterations (default: 1000)
- `regularization`: 'none', 'l1', or 'l2' (default: 'none')
- `lambda_param`: Regularization strength

**Complexity**:
- Time: O(nmk) where k = iterations
- Space: O(n)

**Pros**:
- Simple and interpretable
- Fast to train
- Works well with linear relationships

**Cons**:
- Assumes linear relationship
- Sensitive to outliers
- Not suitable for non-linear patterns

---

### Logistic Regression

**Purpose**: Binary classification

**Usage**:
```python
from src.supervised import LogisticRegression

model = LogisticRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
accuracy = model.score(X_test, y_test)
```

**Parameters**:
- `learning_rate`: Step size (default: 0.01)
- `iterations`: Training iterations (default: 1000)
- `regularization`: Regularization type
- `lambda_param`: Regularization strength

**Complexity**:
- Time: O(nmk)
- Space: O(n)

**Pros**:
- Probabilistic output
- Fast and scalable
- Works well for linearly separable classes

**Cons**:
- Limited to binary classification (multiclass with one-vs-rest)
- Requires scaling for best performance
- Assumes linear decision boundary

---

### Decision Tree

**Purpose**: Classification and regression

**Usage**:
```python
from src.supervised import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=10, min_samples_split=2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)
```

**Parameters**:
- `max_depth`: Maximum tree depth (default: 10)
- `min_samples_split`: Min samples to split (default: 2)

**Complexity**:
- Time: O(n log n × features) for training
- Space: O(n) for tree storage
- Prediction: O(log n)

**Pros**:
- Interpretable
- Handles non-linear relationships
- No scaling needed
- Works with both numeric and categorical data

**Cons**:
- Prone to overfitting
- Unstable (small data changes → large tree changes)
- Biased toward dominant classes

---

### Random Forest

**Purpose**: Classification and regression (ensemble)

**Usage**:
```python
from src.supervised import RandomForestClassifier

model = RandomForestClassifier(n_trees=10, max_depth=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
importances = model.feature_importance(X_train, y_train)
```

**Parameters**:
- `n_trees`: Number of trees (default: 10)
- `max_depth`: Max tree depth
- `min_samples_split`: Min samples to split
- `random_state`: Random seed

**Complexity**:
- Time: O(n × trees × log n)
- Space: O(trees × n)

**Pros**:
- Reduces overfitting vs single tree
- Provides feature importance
- Parallelizable
- Robust to outliers

**Cons**:
- Computationally expensive
- Less interpretable than single tree
- Memory intensive with many trees

---

### Support Vector Machine (SVM)

**Purpose**: Binary classification

**Usage**:
```python
from src.supervised import SVM

model = SVM(learning_rate=0.01, iterations=1000, lambda_param=0.01)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
decision_values = model.decision_function(X_test)
```

**Parameters**:
- `learning_rate`: Learning rate (default: 0.01)
- `iterations`: Training iterations
- `lambda_param`: Regularization strength

**Complexity**:
- Time: O(nmk)
- Space: O(n)

**Pros**:
- Works well in high dimensions
- Memory efficient
- Effective with clear margins

**Cons**:
- Requires scaling
- Slow with large datasets
- Labels must be -1 and 1
- Requires C parameter tuning

---

## Unsupervised Learning

### K-Means Clustering

**Purpose**: Partition data into k clusters

**Usage**:
```python
from src.unsupervised import KMeans

model = KMeans(k=3, max_iterations=100)
labels = model.fit_predict(X)
predictions = model.predict(X_new)
```

**Parameters**:
- `k`: Number of clusters (default: 3)
- `max_iterations`: Max iterations
- `random_state`: Random seed

**Complexity**:
- Time: O(nkdi) where i = iterations
- Space: O(nk)

**Pros**:
- Simple and fast
- Scalable
- Converges quickly

**Cons**:
- Must specify k beforehand
- Sensitive to initialization
- Converges to local minima
- Requires numeric features

---

### K-Means++

**Purpose**: K-Means with better initialization

**Usage**:
```python
from src.unsupervised import KMeansPlusPlus

model = KMeansPlusPlus(k=3, random_state=42)
labels = model.fit_predict(X)
```

**Improvement over K-Means**:
- Better initial centroids
- Faster convergence
- Better final clustering quality

---

### Principal Component Analysis (PCA)

**Purpose**: Dimensionality reduction via variance maximization

**Usage**:
```python
from src.unsupervised import PCA

model = PCA(n_components=2)
X_reduced = model.fit_transform(X)
X_reconstructed = model.inverse_transform(X_reduced)
variance_ratio = model.explained_variance_ratio()
```

**Parameters**:
- `n_components`: Number of components to keep

**Complexity**:
- Time: O(n × p²) for eigendecomposition
- Space: O(np)

**Pros**:
- Reduces computational complexity
- Removes noise
- Visualizes high-dimensional data
- Removes correlated features

**Cons**:
- Components not interpretable
- Assumes linear relationships
- Sensitive to scaling
- Information loss

---

### Hierarchical Clustering

**Purpose**: Build hierarchy of clusters (dendrogram)

**Usage**:
```python
from src.unsupervised import HierarchicalClustering

model = HierarchicalClustering(n_clusters=3, linkage='complete')
labels = model.fit_predict(X)
```

**Parameters**:
- `n_clusters`: Final number of clusters
- `linkage`: 'single', 'complete', or 'average'

**Complexity**:
- Time: O(n²) to O(n³)
- Space: O(n²)

**Linkage Methods**:
- **Single**: Minimum distance (tend to chain)
- **Complete**: Maximum distance (compact clusters)
- **Average**: Mean distance (balanced)

**Pros**:
- Produces dendrogram (interpretable)
- Doesn't require k specification
- Deterministic (no local minima)

**Cons**:
- Computationally expensive
- Sensitive to noise
- Once merged, can't unmerge

---

### Isolation Forest

**Purpose**: Anomaly/outlier detection

**Usage**:
```python
from src.unsupervised import IsolationForest

model = IsolationForest(n_trees=100, random_state=42)
model.fit(X_train)
scores = model.decision_function(X_test)
predictions = model.predict(X_test)  # 1=anomaly, -1=normal
```

**Parameters**:
- `n_trees`: Number of isolation trees
- `sample_size`: Size of sample per tree
- `max_depth`: Maximum tree depth

**Complexity**:
- Time: O(trees × n log n)
- Space: O(trees × n)

**Pros**:
- No distance calculation needed
- Efficient for high dimensions
- Handles multivariate anomalies
- No parameter tuning needed

**Cons**:
- Assumes anomalies are isolated
- Performance depends on tree quality
- Less effective with very rare anomalies

---

## Neural Networks

### Multi-Layer Perceptron (MLP)

**Purpose**: General-purpose classification/regression

**Usage**:
```python
from src.neural_networks import MLP, one_hot_encode

y_train_encoded = one_hot_encode(y_train, n_classes=3)
model = MLP([784, 128, 64, 10], learning_rate=0.01, iterations=100)
model.fit(X_train, y_train_encoded)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

**Architecture**:
- Input layer
- Hidden layers (customizable)
- Output layer

**Complexity**:
- Time: O(n × layers × units²)
- Space: O(Σ(layer_i × layer_i+1))

**Pros**:
- Universal approximator
- Flexible architecture
- Works for any problem

**Cons**:
- Requires tuning (learning rate, iterations, layers)
- Prone to overfitting
- Slow convergence
- Needs scaled features

---

### LSTM (Long Short-Term Memory)

**Purpose**: Sequence modeling with long-term dependencies

**Usage**:
```python
from src.neural_networks import LSTM

lstm = LSTM(input_dim=8, hidden_dim=16)
h, c = lstm.forward(X_sequence)  # X: (seq_len, batch, features)
```

**Key Features**:
- Forget gate: Decides what to forget
- Input gate: Decides what to add
- Output gate: Decides what to output
- Cell state: Memory

**Complexity**:
- Time: O(seq_len × batch × hidden²)
- Space: O(seq_len × batch × hidden)

**Pros**:
- Captures long-term dependencies
- Mitigates vanishing gradient
- Effective for sequences

**Cons**:
- Computationally expensive
- Hard to interpret
- Requires lots of data
- Slow inference

---

### CNN (Convolutional Neural Network)

**Purpose**: Image classification and feature extraction

**Usage**:
```python
from src.neural_networks import SimpleCNN

cnn = SimpleCNN(output_dim=10)
features = cnn.predict(X_images)  # X: (batch, height, width, channels)
```

**Key Components**:
- Convolutional layers: Extract features
- Pooling layers: Reduce dimensions
- Fully connected: Classification

**Complexity**:
- Time: O(batch × channels × k² × h × w)
- Space: O(batch × channels × h × w)

**Pros**:
- Parameter sharing reduces parameters
- Captures spatial relationships
- Translation invariant
- State-of-the-art for images

**Cons**:
- Computationally expensive
- Requires large datasets
- Hard to interpret
- Not suitable for non-image data

---

## Choosing an Algorithm

### For Regression
- **Linear data**: Linear Regression
- **Non-linear data**: Decision Tree, Random Forest
- **Interpretability priority**: Linear Regression
- **Performance priority**: Random Forest

### For Classification
- **Linearly separable**: Logistic Regression, SVM
- **Non-linear**: Decision Tree, Random Forest
- **Complex patterns**: Neural Networks
- **Speed priority**: Logistic Regression
- **Accuracy priority**: Random Forest, Neural Networks

### For Unsupervised
- **Clustering**: K-Means, Hierarchical Clustering
- **Dimensionality reduction**: PCA
- **Anomaly detection**: Isolation Forest
- **Interpretability**: Hierarchical Clustering

---

## Quick Comparison Table

| Algorithm | Type | Time | Space | Scaling | Interpretable |
|-----------|------|------|-------|---------|---------------|
| Linear Regression | Supervised | O(nmk) | O(n) | Yes | Yes |
| Logistic Regression | Supervised | O(nmk) | O(n) | Yes | Yes |
| Decision Tree | Supervised | O(n log n) | O(n) | No | Yes |
| Random Forest | Supervised | O(nt log n) | O(tn) | No | Partial |
| SVM | Supervised | O(nmk) | O(n) | Yes | No |
| K-Means | Unsupervised | O(nkdi) | O(nk) | Yes | Partial |
| PCA | Unsupervised | O(np²) | O(np) | Yes | No |
| Hierarchical | Unsupervised | O(n²) | O(n²) | No | Yes |
| Isolation Forest | Unsupervised | O(tn log n) | O(tn) | Yes | No |
| MLP | Supervised | O(n×l×u²) | O(l×u) | Yes | No |
