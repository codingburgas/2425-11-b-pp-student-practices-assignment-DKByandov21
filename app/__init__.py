"""
Инициализира Flask разширенията и регистрира blueprint-и за модулна организация.
"""

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_migrate import Migrate

# Инициализиране на разширенията
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()


def create_app(config_name=None):
    """
    Създава и конфигурира Flask приложение.
    """
    app = Flask(__name__)
    
    # Зареждане на конфигурация
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'default')
    
    from instance.config import config
    app.config.from_object(config[config_name])
    
    # Създаване на папката за качени файлове, ако не съществува
    upload_folder = app.config.get('UPLOAD_FOLDER')
    if upload_folder and not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    # Инициализиране на разширенията с приложението
    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)
    
    # Конфигуриране на login manager
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    login_manager.login_message = 'Please log in to access this page.'
    
    # Регистриране на blueprint-и
    from app.routes.auth import auth as auth_blueprint
    from app.routes.main import main as main_blueprint
    from app.routes.admin import admin as admin_blueprint
    
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    app.register_blueprint(main_blueprint)
    app.register_blueprint(admin_blueprint, url_prefix='/admin')
    
    # Импортиране на моделите, за да се регистрират със SQLAlchemy
    from app.models import user, prediction, feedback
    
    # Зареждане на потребител за Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        """Зарежда потребител по ID за Flask-Login."""
        from app.models.user import User
        return User.query.get(int(user_id))
    
    # Обработчици на грешки
    @app.errorhandler(404)
    def not_found_error(error):
        from flask import render_template
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        from flask import render_template
        db.session.rollback()
        return render_template('errors/500.html'), 500
    
    return app
