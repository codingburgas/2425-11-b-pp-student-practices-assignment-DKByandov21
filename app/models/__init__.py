
"""
Database models package for Shape Classifier.

This package contains all SQLAlchemy models for the application.
All models are imported here to ensure they're registered with SQLAlchemy.
"""

from app.models.user import User
from app.models.prediction import Prediction
from app.models.feedback import Feedback

__all__ = ['User', 'Prediction', 'Feedback']
