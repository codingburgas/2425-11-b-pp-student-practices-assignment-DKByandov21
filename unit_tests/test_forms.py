
"""
Tests for WTForms validation.
"""

import pytest
import sys
import os
# Add the parent directory of the current script (which contains 'app') to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.forms import RegistrationForm, LoginForm, FeedbackForm

class TestRegistrationForm:
    """Test registration form validation."""
    
    def test_valid_registration(self, app):
        """Test valid registration form."""
        with app.app_context():
            form = RegistrationForm(
                username='testuser',
                email='test@example.com',
                password='password123',
                confirm_password='password123'
            )
            assert form.validate() is True
    
    def test_password_mismatch(self, app):
        """Test password confirmation mismatch."""
        with app.app_context():
            form = RegistrationForm(
                username='testuser',
                email='test@example.com',
                password='password123',
                confirm_password='different'
            )
            assert form.validate() is False
            assert 'Field must be equal to password' in str(form.confirm_password.errors)
    
    def test_invalid_email(self, app):
        """Test invalid email format."""
        with app.app_context():
            form = RegistrationForm(
                username='testuser',
                email='invalid-email',
                password='password123',
                confirm_password='password123'
            )
            assert form.validate() is False
            assert 'Invalid email address' in str(form.email.errors)
    
    def test_short_password(self, app):
        """Test password too short."""
        with app.app_context():
            form = RegistrationForm(
                username='testuser',
                email='test@example.com',
                password='123',
                confirm_password='123'
            )
            assert form.validate() is False


class TestLoginForm:
    """Test login form validation."""
    
    def test_valid_login(self, app):
        """Test valid login form."""
        with app.app_context():
            form = LoginForm(
                email='test@example.com',
                password='password123'
            )
            assert form.validate() is True
    
    def test_missing_email(self, app):
        """Test missing email."""
        with app.app_context():
            form = LoginForm(
                email='',
                password='password123'
            )
            assert form.validate() is False


class TestFeedbackForm:
    """Test feedback form validation."""
    
    def test_valid_feedback(self, app):
        """Test valid feedback form."""
        with app.app_context():
            form = FeedbackForm(
                rating=5,
                comment='Great app!',
                is_public=True
            )
            assert form.validate() is True
    
    def test_missing_rating(self, app):
        """Test missing rating."""
        with app.app_context():
            form = FeedbackForm(
                comment='Great app!',
                is_public=True
            )
            assert form.validate() is False
    
    def test_long_comment(self, app):
        """Test comment too long."""
        with app.app_context():
            long_comment = 'x' * 1001  # Exceeds 1000 char limit
            form = FeedbackForm(
                rating=5,
                comment=long_comment,
                is_public=True
            )
            assert form.validate() is False
