
"""
Routes package for Shape Classifier.

This package contains all Flask blueprints for the application.
Blueprints are organized by functionality:
- auth: Authentication routes (login, register, logout)
- main: Main application routes (profile, predict, feedback)
- admin: Administrative routes (dashboard, user management)
"""

from app.routes.auth import auth
from app.routes.main import main
from app.routes.admin import admin

__all__ = ['auth', 'main', 'admin']
