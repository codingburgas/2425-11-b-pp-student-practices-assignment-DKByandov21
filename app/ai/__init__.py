"""
- perceptron: Собствена имплементация на Персептрон
- logistic_regression: Имплементация на Логистична Регресия
- model_utils: Помощни функции за запазване и зареждане на модели
- synthetic_dataset: Инструменти за генериране на синтетични набори от данни
- metrics: Метрики за оценка и помощни функции
"""

from .perceptron import Perceptron
from .logistic_regression import LogisticRegression
from .model_utils import save_model, load_model
from .synthetic_dataset import create_synthetic_dataset
from .metrics import ModelMetrics, InformationGain

# Създаване на удобни функции, които обвиват статичните методи на ModelMetrics
def accuracy_score(y_true, y_pred):
    """Тук изчисляване на точност (accuracy)."""
    return ModelMetrics.accuracy(y_true, y_pred)

def precision_score(y_true, y_pred):
    """Тук изчисляване на прецизност (precision)."""
    return ModelMetrics.precision(y_true, y_pred)

def recall_score(y_true, y_pred):
    """Тук изчисляване на чувствителност (recall)."""
    return ModelMetrics.recall(y_true, y_pred)

def f1_score(y_true, y_pred):
    """Тук изчисляване на F1 резултат."""
    return ModelMetrics.f1_score(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    """Тук изчисляване на матрица на объркванията."""
    return ModelMetrics.confusion_matrix(y_true, y_pred)

def log_loss(y_true, y_pred_proba):
    """Тук изчисляване на лог загуба (log loss)."""
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
