"""
Flask Application Factory for Shape Classifier.

This module contains the application factory pattern implementation for the
Shape Classifier web application. It initializes Flask extensions and
registers blueprints for modular organization.

The application uses a factory pattern to allow for different configurations
(development, testing, production) and to enable proper testing.
"""

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate
from config import Config

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()


def create_app(config_class=Config):
    """
    Create and configure a Flask application instance.
    
    This function implements the application factory pattern, creating
    a Flask app with the specified configuration and initializing all
    extensions and blueprints.
    
    Args:
        config_class: Configuration class to use for the application.
                      Defaults to Config from config.py.
    
    Returns:
        Flask: Configured Flask application instance.
    
    Example:
        >>> app = create_app()
        >>> app.run(debug=True)
    """
    # Create Flask application instance
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions with the application
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    
    # Configure login manager settings
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    
    # Import and register blueprints for modular organization
    from app.routes.auth import auth
    from app.routes.main import main
    from app.routes.admin import admin
    app.register_blueprint(auth, url_prefix="/auth", template_folder="templates")
    app.register_blueprint(main, url_prefix="/", template_folder="templates")
    app.register_blueprint(admin, url_prefix="/admin", template_folder="templates")
    
    # Import models to ensure they're registered with SQLAlchemy
    # This is necessary for migrations and database operations
    from app.models.user import User
    from app.models.prediction import Prediction
    
    # User loader function for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        """
        Load user by ID for Flask-Login.
        
        This function is called by Flask-Login to load a user from the
        database when needed for session management.
        
        Args:
            user_id (str): The user ID as a string.
        
        Returns:
            User: User object if found, None otherwise.
        """
        return User.query.get(int(user_id))
    
    return app 