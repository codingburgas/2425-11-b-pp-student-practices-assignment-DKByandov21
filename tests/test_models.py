
"""
Tests for database models.
"""

import pytest
from datetime import datetime
from app import db
from app.models.user import User
from app.models.prediction import Prediction
from app.models.feedback import Feedback


class TestUser:
    """Test User model."""
    
    def test_password_hashing(self, app):
        """Test password hashing and verification."""
        with app.app_context():
            user = User(username='test', email='test@example.com')
            user.set_password('secret')
            
            assert user.password_hash is not None
            assert user.password_hash != 'secret'
            assert user.check_password('secret') is True
            assert user.check_password('wrong') is False
    
    def test_user_roles(self, app):
        """Test user role functionality."""
        with app.app_context():
            user = User(username='user', email='user@example.com')
            admin = User(username='admin', email='admin@example.com', role='admin')
            
            assert user.is_admin() is False
            assert admin.is_admin() is True
    
    def test_user_active_status(self, app):
        """Test user active status."""
        with app.app_context():
            user = User(username='test', email='test@example.com')
            assert user.is_active() is True
            
            user.active = False
            assert user.is_active() is False
    
    def test_profile_picture_url(self, app):
        """Test profile picture URL generation."""
        with app.app_context():
            user = User(username='test', email='test@example.com')
            
            # Test default URL
            default_url = user.get_profile_picture_url()
            assert 'ui-avatars.com' in default_url
            assert 'test' in default_url
            
            # Test custom picture
            user.profile_picture = 'test.jpg'
            custom_url = user.get_profile_picture_url()
            assert '/static/uploads/test.jpg' == custom_url


class TestPrediction:
    """Test Prediction model."""
    
    def test_prediction_creation(self, app, user):
        """Test prediction creation."""
        with app.app_context():
            prediction = Prediction(
                user_id=user.id,
                filename='test.jpg',
                prediction='Circle',
                confidence=0.95
            )
            db.session.add(prediction)
            db.session.commit()
            
            assert prediction.id is not None
            assert prediction.user_id == user.id
            assert prediction.prediction == 'Circle'
            assert prediction.confidence == 0.95
            assert prediction.created_at is not None


class TestFeedback:
    """Test Feedback model."""
    
    def test_feedback_creation(self, app, user):
        """Test feedback creation."""
        with app.app_context():
            feedback = Feedback(
                user_id=user.id,
                rating=5,
                comment='Great app!',
                is_public=True
            )
            db.session.add(feedback)
            db.session.commit()
            
            assert feedback.id is not None
            assert feedback.user_id == user.id
            assert feedback.rating == 5
            assert feedback.comment == 'Great app!'
            assert feedback.is_public is True
            assert feedback.created_at is not None
