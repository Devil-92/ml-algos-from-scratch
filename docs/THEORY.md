# Mathematical Theory Behind ML Algorithms

This document provides mathematical foundations for all implemented algorithms.

## Linear Algebra Fundamentals

### Matrix Operations
- **Dot Product**: `a · b = Σ aᵢbᵢ`
- **Matrix Multiplication**: `(AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ`
- **Transpose**: `(Aᵀ)ᵢⱼ = Aⱼᵢ`
- **Inverse**: `A⁻¹A = I`
- **Eigenvalues/Eigenvectors**: `Av = λv`

## Supervised Learning

### Linear Regression

**Objective**: Minimize Mean Squared Error
```
L = (1/2m) Σ(ŷᵢ - yᵢ)²
where ŷᵢ = w₀ + w₁x₁ + ... + wₙxₙ
```

**Gradient Descent Update**:
```
w := w - α(∂L/∂w)
∂L/∂w = (1/m) Xᵀ(ŷ - y)
```

**Normal Equation** (Closed-form):
```
w = (XᵀX)⁻¹Xᵀy
```

**Complexity**: 
- Gradient Descent: O(nmk) per iteration
- Normal Equation: O(n³)

### Logistic Regression

**Sigmoid Activation**:
```
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Cross-Entropy Loss**:
```
L = -(1/m) Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Gradient**:
```
∂L/∂w = (1/m) Xᵀ(ŷ - y)
```

### Decision Trees

**Gini Impurity**:
```
G = 1 - Σ pᵢ²
where pᵢ is proportion of class i
```

**Information Gain**:
```
IG = G(parent) - Σ(n_child/n_parent) G(child)
```

**Splitting**: Choose feature/threshold maximizing information gain

**Complexity**: O(nm log m) where m is number of features

### Random Forest

**Ensemble Method**:
1. Bootstrap sampling: sample n points with replacement
2. Build decision tree on each sample
3. Aggregate predictions: majority voting

**Bias-Variance Trade-off**:
- Single tree: High variance, Low bias
- Forest: Lower variance, Slightly higher bias

### Support Vector Machine (SVM)

**Objective**:
```
minimize: (1/2)||w||² + C Σ ξᵢ
subject to: yᵢ(wᵀφ(xᵢ) + b) ≥ 1 - ξᵢ
```

**Hinge Loss**:
```
L = max(0, 1 - yᵢf(xᵢ))
```

**Optimization**: Stochastic Gradient Descent

## Unsupervised Learning

### K-Means Clustering

**Objective**: Minimize Within-Cluster Sum of Squares
```
WCSS = Σ Σ ||xᵢ - cⱼ||²
```

**Algorithm**:
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids as mean of cluster points
4. Repeat until convergence

**Complexity**: O(nkdi) where d is dimensions, i is iterations

### K-Means++

**Improved Initialization**:
1. Choose first centroid randomly
2. Choose subsequent centroids with probability ∝ distance²
3. Reduces likelihood of poor local minimum

### Principal Component Analysis (PCA)

**Objective**: Find directions of maximum variance

**Steps**:
1. Center data: X' = X - μ
2. Compute covariance: Cov = (1/n)X'ᵀX'
3. Eigendecomposition: Cov = UΛUᵀ
4. Project: X_reduced = X'U_{:k}

**Variance Explained**:
```
variance_ratio = λᵢ / Σλⱼ
```

**Complexity**: O(n × p²) for eigendecomposition

### Hierarchical Clustering

**Linkage Methods**:
- **Single**: min distance between any two points
- **Complete**: max distance between any two points
- **Average**: mean distance between all pairs

**Algorithm**:
1. Start with n clusters (each point)
2. Repeatedly merge closest clusters
3. Stop at desired k clusters

**Complexity**: O(n²)

### Anomaly Detection (Isolation Forest)

**Key Insight**: Anomalies isolated faster in random trees

**Algorithm**:
1. Build t random isolation trees
2. Path length l to isolate point x
3. Anomaly score: s(x) = 2^(-l/c(n))
4. High score = anomaly

**Complexity**: O(t × n log n)

## Neural Networks

### Multi-Layer Perceptron (MLP)

**Forward Pass**:
```
aˡ = σ(Wˡaˡ⁻¹ + bˡ)
```

**Backpropagation**:
```
∂L/∂w = (∂L/∂a)(∂a/∂z)(∂z/∂w)
w := w - α(∂L/∂w)
```

**Activation Functions**:
- ReLU: max(0, x)
- Sigmoid: 1/(1+e⁻ˣ)
- Tanh: (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)

### Recurrent Neural Network (RNN)

**Standard RNN**:
```
hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + bₕ)
yₜ = Wₕyₕₜ + by
```

**Problem**: Vanishing/exploding gradients

### LSTM (Long Short-Term Memory)

**Gates**:
```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)        // Forget
iₜ = σ(Wᵢ·[hₜ₋₁, xₜ] + bᵢ)        // Input
C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)     // Cell candidate
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ          // Cell state update
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)        // Output
hₜ = oₜ ⊙ tanh(Cₜ)                 // Hidden state
```

**Advantages**:
- Learns long-term dependencies
- Mitigates vanishing gradient problem

### Convolutional Neural Network (CNN)

**Convolution Operation**:
```
out[i,j] = Σ kernel ⊙ input[i:i+kh, j:j+kw] + bias
```

**Pooling**:
```
max_pool[i,j] = max(window[i:i+ps, j:j+ps])
```

**Advantages**:
- Parameter sharing reduces parameters
- Local connectivity captures spatial patterns
- Translation invariance

## Regularization Techniques

### L1 Regularization (Lasso)
```
Loss = MSE + λ Σ|w|
```
Feature selection: some weights → 0

### L2 Regularization (Ridge)
```
Loss = MSE + (λ/2) Σ w²
```
Prevents large weights

### Early Stopping
Stop training when validation loss increases

## Optimization Algorithms

### Gradient Descent
```
w := w - α∇L
```

### Stochastic Gradient Descent (SGD)
```
w := w - α∇L(xᵢ, yᵢ)  // Single sample
```

### Momentum
```
v := βv + ∇L
w := w - αv
```

### Adam
```
Combines momentum and adaptive learning rates
```

## Performance Metrics

### Classification
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2(Precision × Recall) / (Precision + Recall)

### Regression
- **MSE**: (1/n) Σ(ŷᵢ - yᵢ)²
- **RMSE**: √MSE
- **R²**: 1 - (SS_res / SS_tot)

### Clustering
- **Inertia**: Σ ||xᵢ - cⱼ||²
- **Silhouette Score**: (b-a) / max(a,b)
- **Davies-Bouldin Index**: Avg(cluster separation)

## References

- [Bishop, 2006] - Pattern Recognition and Machine Learning
- [Goodfellow et al., 2016] - Deep Learning
- [Murphy, 2012] - Machine Learning: A Probabilistic Perspective
- [Hastie et al., 2009] - Elements of Statistical Learning
