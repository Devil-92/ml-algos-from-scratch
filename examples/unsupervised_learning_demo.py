"""
Unsupervised Learning Examples - Demonstrating all unsupervised algorithms.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')

from src.unsupervised import (KMeans, KMeansPlusPlus, PCA,
                              HierarchicalClustering, IsolationForest)


def main():
    """Run unsupervised learning examples."""
    
    print("=" * 70)
    print("ML ALGORITHMS FROM SCRATCH - UNSUPERVISED LEARNING EXAMPLES")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Create synthetic data with clusters
    X = np.vstack([
        np.random.randn(40, 2) + [0, 0],
        np.random.randn(40, 2) + [5, 5],
        np.random.randn(40, 2) + [10, 0]
    ])
    
    print("\n1. K-MEANS CLUSTERING")
    print("-" * 70)
    model = KMeans(k=3, random_state=42)
    labels = model.fit_predict(X)
    print(f"Cluster centroids:\n{model.centroids}")
    print(f"Within-cluster sum of squares: {model.inertia_:.2f}")
    
    print("\n2. K-MEANS++ (BETTER INITIALIZATION)")
    print("-" * 70)
    model_pp = KMeansPlusPlus(k=3, random_state=42)
    labels_pp = model_pp.fit_predict(X)
    print(f"K-means++ inertia: {model_pp.inertia_:.2f}")
    print(f"Improvement: {model.inertia_ - model_pp.inertia_:.2f}")
    
    print("\n3. HIERARCHICAL CLUSTERING")
    print("-" * 70)
    for linkage in ['single', 'complete', 'average']:
        model = HierarchicalClustering(n_clusters=3, linkage=linkage)
        labels = model.fit_predict(X)
        print(f"  {linkage.upper():8s} linkage - Distances: {model.distances_[-3:]}")
    
    print("\n4. PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("-" * 70)
    # Create high-dimensional data
    X_high_dim = np.random.randn(100, 20)
    
    model = PCA(n_components=2)
    X_reduced = model.fit_transform(X_high_dim)
    
    print(f"Original dimensions: {X_high_dim.shape[1]}")
    print(f"Reduced dimensions: {X_reduced.shape[1]}")
    print(f"Explained variance ratio: {model.explained_variance_ratio().round(3)}")
    print(f"Cumulative explained variance: {model.cumulative_explained_variance().round(3)}")
    
    # Reconstruction error
    X_reconstructed = model.inverse_transform(X_reduced)
    mse = np.mean((X_high_dim - X_reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse:.4f}")
    
    print("\n5. ISOLATION FOREST (ANOMALY DETECTION)")
    print("-" * 70)
    # Add anomalies
    X_with_anomalies = np.vstack([
        np.random.randn(100, 2),
        np.random.uniform(-10, 10, (5, 2))
    ])
    
    model = IsolationForest(n_trees=100, random_state=42)
    model.fit(X_with_anomalies)
    
    scores = model.decision_function(X_with_anomalies)
    predictions = model.predict(X_with_anomalies)
    
    n_anomalies = np.sum(predictions == 1)
    print(f"Anomalies detected: {n_anomalies}")
    print(f"Anomaly scores (last 5): {scores[-5:].round(3)}")
    print(f"Normal scores (first 5): {scores[:5].round(3)}")
    
    print("\n" + "=" * 70)
    print("All unsupervised learning algorithms demonstrated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
