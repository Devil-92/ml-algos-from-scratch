"""
Supervised Learning Examples - Demonstrating all supervised algorithms.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/mamunchowdhury/Desktop/ml-algos-from-scratch.worktrees/agents-ml-algorithms-from-scratch')

from src.supervised import (LinearRegression, LogisticRegression, DecisionTreeClassifier,
                           RandomForestClassifier, SVM)
from src.utils import train_test_split, standardize


def main():
    """Run supervised learning examples."""
    
    print("=" * 70)
    print("ML ALGORITHMS FROM SCRATCH - SUPERVISED LEARNING EXAMPLES")
    print("=" * 70)
    
    # Generate synthetic datasets
    np.random.seed(42)
    
    # Regression dataset
    print("\n1. LINEAR REGRESSION")
    print("-" * 70)
    X_reg = np.random.randn(150, 3)
    y_reg = 2 * X_reg[:, 0] + 3 * X_reg[:, 1] - X_reg[:, 2] + np.random.randn(150) * 0.5
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2)
    
    model = LinearRegression(learning_rate=0.1, iterations=500)
    model.fit(X_train, y_train)
    r2_score = model.score(X_test, y_test)
    print(f"Linear Regression R² Score: {r2_score:.4f}")
    print(f"Sample predictions: {model.predict(X_test[:3]).round(2)}")
    
    # Classification datasets
    print("\n2. LOGISTIC REGRESSION")
    print("-" * 70)
    X_clf = np.random.randn(200, 3)
    y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2)
    
    model = LogisticRegression(learning_rate=0.1, iterations=500)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    print(f"Sample probabilities: {model.predict_proba(X_test[:3]).round(3)}")
    
    print("\n3. DECISION TREE")
    print("-" * 70)
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    
    print("\n4. RANDOM FOREST")
    print("-" * 70)
    model = RandomForestClassifier(n_trees=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    importances = model.feature_importance(X_train, y_train)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print(f"Feature Importances: {importances.round(3)}")
    
    print("\n5. SUPPORT VECTOR MACHINE (SVM)")
    print("-" * 70)
    model = SVM(learning_rate=0.01, iterations=500)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"SVM Accuracy: {accuracy:.4f}")
    
    print("\n" + "=" * 70)
    print("All supervised learning algorithms demonstrated successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
