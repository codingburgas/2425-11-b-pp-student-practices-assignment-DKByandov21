from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models.user import User
from app.forms import RegistrationForm, LoginForm

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    # Ако потребителят вече е логнат, го пренасочи към профила
    if current_user.is_authenticated:
        return redirect(url_for('main.profile'))
    
    # Създаване на форма за регистрация
    form = RegistrationForm()
    
    # Ако формата е изпратена успешно и минава валидацията
    if form.validate_on_submit():
        # Създаване на нов потребител
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    
    # Рендиране на шаблона за регистрация с формата
    return render_template('auth/register.html', form=form)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    # Ако потребителят вече е логнат, го пренасочи към профила
    if current_user.is_authenticated:
        return redirect(url_for('main.profile'))
    
    # Създаване на форма за вход
    form = LoginForm()
    
    # Обработка на подадена форма
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        
        # Проверка на парола и активност
        if user and user.check_password(form.password.data) and user.is_active():
            login_user(user)
            return redirect(url_for('main.profile'))
        elif user and not user.is_active():
            flash('Account has been deactivated. Contact administrator.', 'error')
        else:
            flash('Invalid email or password.', 'danger')
    
    # Рендиране на шаблона за вход с формата
    return render_template('auth/login.html', form=form)

@auth.route('/logout')
@login_required
def logout():
    # Излизане от акаунта
    logout_user()
    flash('Logged out.', 'info')
    return redirect(url_for('main.index'))
