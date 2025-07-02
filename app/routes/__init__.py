
"""
Blueprint-ите са организирани според функционалността:
- auth: Маршрути за автентикация (вход, регистрация, изход)
- main: Основни маршрути на приложението (профил, предсказване, обратна връзка)
- admin: Административни маршрути (табло, управление на потребители)
"""

from app.routes.auth import auth
from app.routes.main import main
from app.routes.admin import admin

__all__ = ['auth', 'main', 'admin']
