"""
Main entry point for the Shape Classifier Flask application.

This module creates the Flask application using the application factory
pattern and runs it with appropriate configuration.
"""

import os
from app import create_app, db
from app.models.user import User
from app.models.prediction import Prediction
from app.models.feedback import Feedback
from flask_migrate import upgrade


def deploy():
    """Run deployment tasks."""
    app = create_app()
    app.app_context().push()

    # Create database tables
    db.create_all()

    # Migrate database to latest revision
    upgrade()


# Create application instance
app = create_app()


@app.shell_context_processor
def make_shell_context():
    """Make database models available in shell context."""
    return {
        'db': db,
        'User': User,
        'Prediction': Prediction,
        'Feedback': Feedback
    }


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
