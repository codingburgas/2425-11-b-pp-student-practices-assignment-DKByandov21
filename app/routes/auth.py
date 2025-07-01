from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models.user import User
from app.forms import RegistrationForm, LoginForm
from app.utils import send_confirmation_email
from datetime import datetime, timedelta

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.profile'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        
        # Send confirmation email
        send_confirmation_email(user)
        flash('Registration successful! Please check your email to confirm your account before logging in.', 'success')
        return redirect(url_for('auth.unconfirmed'))
    return render_template('auth/register.html', form=form)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.profile'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data) and user.is_active():
            if not user.is_confirmed():
                flash('Please confirm your email address first.', 'warning')
                return redirect(url_for('auth.unconfirmed'))
            login_user(user)
            return redirect(url_for('main.profile'))
        elif user and not user.is_active():
            flash('Account has been deactivated. Contact administrator.', 'error')
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('auth/login.html', form=form)

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'info')
    return redirect(url_for('main.index'))

@auth.route('/confirm/<token>')
def confirm_email(token):
    """Confirm email with token"""
    if current_user.is_authenticated and current_user.is_confirmed():
        return redirect(url_for('main.profile'))
    
    email = User.confirm_token(token)
    if email is None:
        flash('The confirmation link is invalid or has expired.', 'danger')
        return redirect(url_for('auth.unconfirmed'))
    
    user = User.query.filter_by(email=email).first_or_404()
    if user.is_confirmed():
        flash('Account already confirmed. Please login.', 'success')
    else:
        user.confirm_email()
        flash('You have confirmed your account. Thank you!', 'success')
    
    return redirect(url_for('auth.login'))

@auth.route('/unconfirmed')
def unconfirmed():
    """Page for unconfirmed users"""
    if current_user.is_anonymous or current_user.is_confirmed():
        return redirect(url_for('main.index'))
    return render_template('auth/unconfirmed.html')

@auth.route('/resend-confirmation')
def resend_confirmation():
    """Resend confirmation email"""
    if current_user.is_anonymous:
        return redirect(url_for('auth.login'))
    
    if current_user.is_confirmed():
        return redirect(url_for('main.profile'))
    
    # Check if we've sent a confirmation email recently (cooldown)
    last_sent = request.cookies.get('last_confirmation_sent')
    if last_sent:
        last_sent_time = datetime.fromisoformat(last_sent)
        if datetime.now() - last_sent_time < timedelta(minutes=1):
            flash('Please wait before requesting another confirmation email.', 'warning')
            return redirect(url_for('auth.unconfirmed'))
    
    send_confirmation_email(current_user)
    flash('A new confirmation email has been sent to your email address.', 'info')
    
    # Set cooldown cookie
    response = redirect(url_for('auth.unconfirmed'))
    response.set_cookie('last_confirmation_sent', datetime.now().isoformat(), max_age=60)
    return response 