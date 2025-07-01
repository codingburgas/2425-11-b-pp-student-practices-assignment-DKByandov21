
"""
AI package for Shape Classifier.

This package contains machine learning models and utilities:
- perceptron: Custom Perceptron implementation
- logistic_regression: Logistic Regression implementation
- model_utils: Model persistence and loading utilities
- synthetic_dataset: Dataset generation utilities
- metrics: Evaluation metrics and utilities
"""

from .perceptron import Perceptron
from .logistic_regression import LogisticRegression
from .model_utils import save_model, load_model
from .synthetic_dataset import create_synthetic_dataset
from .metrics import ModelMetrics, InformationGain

# Create convenience functions that wrap the ModelMetrics static methods
def accuracy_score(y_true, y_pred):
    """Calculate accuracy score."""
    return ModelMetrics.accuracy(y_true, y_pred)

def precision_score(y_true, y_pred):
    """Calculate precision score."""
    return ModelMetrics.precision(y_true, y_pred)

def recall_score(y_true, y_pred):
    """Calculate recall score."""
    return ModelMetrics.recall(y_true, y_pred)

def f1_score(y_true, y_pred):
    """Calculate F1 score."""
    return ModelMetrics.f1_score(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix."""
    return ModelMetrics.confusion_matrix(y_true, y_pred)

def log_loss(y_true, y_pred_proba):
    """Calculate log loss."""
    return ModelMetrics.log_loss(y_true, y_pred_proba)

__all__ = [
    'Perceptron',
    'LogisticRegression', 
    'save_model',
    'load_model',
    'create_synthetic_dataset',
    'ModelMetrics',
    'InformationGain',
    'accuracy_score',
    'precision_score', 
    'recall_score',
    'f1_score',
    'confusion_matrix',
    'log_loss'
]
