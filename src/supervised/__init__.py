"""Supervised learning algorithms."""

from .linear_regression import LinearRegression, LinearRegressionClosedForm
from .logistic_regression import LogisticRegression
from .decision_tree import DecisionTreeClassifier
from .random_forest import RandomForestClassifier
from .svm import SVM

__all__ = [
    'LinearRegression',
    'LinearRegressionClosedForm',
    'LogisticRegression',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'SVM',
]
