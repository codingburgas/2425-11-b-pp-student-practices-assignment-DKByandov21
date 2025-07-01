
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
from app.ai.synthetic_dataset import generate_circle_square_dataset
from app.ai.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

__all__ = [
    'Perceptron',
    'LogisticRegression', 
    'save_model',
    'load_model',
    'generate_circle_square_dataset',
    'accuracy_score',
    'precision_score', 
    'recall_score',
    'f1_score',
    'confusion_matrix'
]
