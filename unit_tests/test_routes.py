"""
Tests for application routes.
"""

import pytest
import io
from PIL import Image
import sys
import os
# Add the parent directory of the current script (which contains 'app') to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import db
from app.models import User


class TestAuth:
    """Test authentication routes."""

    def test_register_get(self, client):
        """Test registration page loads."""
        response = client.get('/auth/register')
        assert response.status_code == 200
        assert b'Register' in response.data

    def test_register_post_valid(self, client, app):
        """Test valid user registration."""
        response = client.post('/auth/register',
                               data={
                                   'username': 'newuser',
                                   'email': 'newuser@example.com',
                                   'password': 'password123',
                                   'confirm_password': 'password123'
                               })

        # Should redirect after successful registration
        assert response.status_code == 302

        # Check user was created
        with app.app_context():
            user = User.query.filter_by(username='newuser').first()
            assert user is not None
            assert user.email == 'newuser@example.com'

    def test_register_duplicate_username(self, client, user):
        """Test registration with duplicate username."""
        response = client.post(
            '/auth/register',
            data={
                'username': 'testuser',  # Same as fixture user
                'email': 'different@example.com',
                'password': 'password123',
                'confirm_password': 'password123'
            })

        assert response.status_code == 200
        assert b'Username already taken' in response.data

    def test_login_valid(self, client, user):
        """Test valid login."""
        response = client.post('/auth/login',
                               data={
                                   'email': 'test@example.com',
                                   'password': 'testpass123'
                               })

        assert response.status_code == 302  # Redirect after login

    def test_login_invalid(self, client):
        """Test invalid login."""
        response = client.post('/auth/login',
                               data={
                                   'email': 'wrong@example.com',
                                   'password': 'wrongpass'
                               })

        assert response.status_code == 200
        assert b'Invalid email or password' in response.data

    def test_logout(self, client, auth):
        """Test logout."""
        auth.register()
        auth.login()

        response = auth.logout()
        assert response.status_code == 302


class TestMain:
    """Test main application routes."""

    def test_index(self, client):
        """Test homepage loads."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Shape Classifier' in response.data

    def test_profile_requires_auth(self, client):
        """Test profile page requires authentication."""
        response = client.get('/profile')
        assert response.status_code == 302  # Redirect to login

    def test_profile_authenticated(self, client, auth):
        """Test profile page when authenticated."""
        auth.register()
        auth.login()

        response = client.get('/profile')
        assert response.status_code == 200
        assert b'testuser' in response.data

    def test_predict_get(self, client, auth):
        """Test prediction page loads."""
        auth.register()
        auth.login()

        response = client.get('/predict')
        assert response.status_code == 200
        assert b'Upload' in response.data


class TestAdmin:
    """Test admin routes."""

    def test_admin_requires_auth(self, client):
        """Test admin pages require authentication."""
        response = client.get('/admin/')
        assert response.status_code == 302

    def test_admin_requires_admin_role(self, client, auth):
        """Test admin pages require admin role."""
        auth.register()
        auth.login()

        response = client.get('/admin/')
        assert response.status_code == 403  # Forbidden

    def test_admin_dashboard_for_admin(self, client, app, admin_user):
        """Test admin dashboard for admin user."""
        with client.session_transaction() as sess:
            sess['user_id'] = admin_user.id
            sess['_fresh'] = True

        # Login as admin
        response = client.post('/auth/login',
                               data={
                                   'email': 'admin@example.com',
                                   'password': 'adminpass123'
                               })

        response = client.get('/admin/')
        assert response.status_code == 200
        assert b'Admin Dashboard' in response.data
