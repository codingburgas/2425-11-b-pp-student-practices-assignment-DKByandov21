
"""
Всички модели се импортират тук, за да се гарантира, че са регистрирани със SQLAlchemy.
"""

from app.models.user import User
from app.models.prediction import Prediction
from app.models.feedback import Feedback

__all__ = ['User', 'Prediction', 'Feedback']
