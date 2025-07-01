
"""
Pytest configuration and shared fixtures for Shape Classifier tests.
"""

import pytest
import tempfile
import os
from app import create_app, db
from app.models.user import User
from app.models.prediction import Prediction
from app.models.feedback import Feedback


@pytest.fixture
def app():
    """Create application for testing."""
    # Create a temporary file for the test database
    db_fd, db_path = tempfile.mkstemp()
    
    app = create_app('testing')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()
    
    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create CLI runner."""
    return app.test_cli_runner()


@pytest.fixture
def auth(client):
    """Authentication helper."""
    class AuthActions:
        def __init__(self, client):
            self._client = client

        def register(self, username='testuser', email='test@example.com', password='testpass123'):
            return self._client.post('/auth/register', data={
                'username': username,
                'email': email,
                'password': password,
                'confirm_password': password
            })

        def login(self, email='test@example.com', password='testpass123'):
            return self._client.post('/auth/login', data={
                'email': email,
                'password': password
            })

        def logout(self):
            return self._client.get('/auth/logout')

    return AuthActions(client)


@pytest.fixture
def user(app):
    """Create test user."""
    with app.app_context():
        user = User(username='testuser', email='test@example.com')
        user.set_password('testpass123')
        db.session.add(user)
        db.session.commit()
        return user


@pytest.fixture
def admin_user(app):
    """Create test admin user."""
    with app.app_context():
        admin = User(username='admin', email='admin@example.com', role='admin')
        admin.set_password('adminpass123')
        db.session.add(admin)
        db.session.commit()
        return admin
