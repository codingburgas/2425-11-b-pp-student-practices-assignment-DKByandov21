
"""
Tests for AI models and algorithms.
"""

import pytest
import numpy as np
import sys
import os
# Add the parent directory of the current script (which contains 'app') to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.ai import Perceptron
from app.ai import LogisticRegression
# Change this line

from app.ai.synthetic_dataset import create_synthetic_dataset as generate_circle_square_dataset
from app.ai import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class TestPerceptron:
    """Test Perceptron implementation."""
    
    def test_perceptron_creation(self):
        """Test perceptron can be created."""
        perceptron = Perceptron(learning_rate=0.1, max_epochs=100)
        assert perceptron.learning_rate == 0.1
        assert perceptron.max_epochs == 100
        assert perceptron.weights is None
        assert perceptron.bias is None
    
    def test_perceptron_training(self):
        """Test perceptron training on simple dataset."""
        # Create simple linearly separable dataset
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])  # AND function
        
        perceptron = Perceptron(learning_rate=0.1, max_epochs=100)
        perceptron.fit(X, y)
        
        assert perceptron.weights is not None
        assert perceptron.bias is not None
        assert len(perceptron.weights) == X.shape[1]
    
    def test_perceptron_prediction(self):
        """Test perceptron prediction."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        
        perceptron = Perceptron(learning_rate=0.1, max_epochs=100)
        perceptron.fit(X, y)
        
        predictions = perceptron.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)


class TestLogisticRegression:
    """Test Logistic Regression implementation."""
    
    def test_logistic_creation(self):
        """Test logistic regression can be created."""
        lr = LogisticRegression(learning_rate=0.01, max_epochs=1000)
        assert lr.learning_rate == 0.01
        assert lr.max_epochs == 1000
        assert lr.weights is None
        assert lr.bias is None
    
    def test_logistic_training(self):
        """Test logistic regression training."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        lr = LogisticRegression(learning_rate=0.1, max_epochs=100)
        lr.fit(X, y)
        
        assert lr.weights is not None
        assert lr.bias is not None
    
    def test_logistic_prediction(self):
        """Test logistic regression prediction."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        lr = LogisticRegression(learning_rate=0.1, max_epochs=100)
        lr.fit(X, y)
        
        predictions = lr.predict(X)
        probabilities = lr.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert len(probabilities) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
        assert all(0 <= prob <= 1 for prob_pair in probabilities for prob in prob_pair)


class TestSyntheticDataset:
    """Test synthetic dataset generation."""
    
    def test_dataset_generation(self):
        """Test circle/square dataset generation."""
        X, y = generate_circle_square_dataset(n_samples=100, image_size=28, noise_level=0.1)
        
        assert X.shape == (100, 28*28)
        assert y.shape == (100,)
        assert all(label in [0, 1] for label in y)
        assert X.min() >= 0
        assert X.max() <= 1


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_accuracy_score(self):
        """Test accuracy calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        accuracy = accuracy_score(y_true, y_pred)
        assert 0 <= accuracy <= 1
        assert accuracy == 0.8  # 4/5 correct
    
    def test_precision_score(self):
        """Test precision calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        precision = precision_score(y_true, y_pred)
        assert 0 <= precision <= 1
    
    def test_recall_score(self):
        """Test recall calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        recall = recall_score(y_true, y_pred)
        assert 0 <= recall <= 1
    
    def test_f1_score(self):
        """Test F1 score calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        f1 = f1_score(y_true, y_pred)
        assert 0 <= f1 <= 1
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
