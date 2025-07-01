
"""
AI package for Shape Classifier.

This package contains machine learning models and utilities:
- perceptron: Custom Perceptron implementation
- logistic_regression: Logistic Regression implementation
- model_utils: Model persistence and loading utilities
- synthetic_dataset: Dataset generation utilities
- metrics: Evaluation metrics and utilities
"""

from app.ai.perceptron import Perceptron
from app.ai.logistic_regression import LogisticRegression
from app.ai.model_utils import save_model, load_model
from app.ai.synthetic_dataset import create_synthetic_dataset
from app.ai.metrics import ModelMetrics

# Create convenience functions that wrap the ModelMetrics static methods
def accuracy_score(y_true, y_pred):
    return ModelMetrics.accuracy(y_true, y_pred)

def precision_score(y_true, y_pred):
    return ModelMetrics.precision(y_true, y_pred)

def recall_score(y_true, y_pred):
    return ModelMetrics.recall(y_true, y_pred)

def f1_score(y_true, y_pred):
    return ModelMetrics.f1_score(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    return ModelMetrics.confusion_matrix(y_true, y_pred)

__all__ = [
    'Perceptron',
    'LogisticRegression', 
    'save_model',
    'load_model',
    'create_synthetic_dataset',
    'accuracy_score',
    'precision_score', 
    'recall_score',
    'f1_score',
    'confusion_matrix'
]
