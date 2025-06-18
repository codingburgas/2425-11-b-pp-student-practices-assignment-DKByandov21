from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from app import db
from app.models.user import User
from app.models.prediction import Prediction
from app.models.feedback import Feedback
from app.ai.perceptron import Perceptron
from app.ai.model_utils import load_model
from app.forms import ProfileUpdateForm, PasswordChangeForm, FeedbackForm

main = Blueprint('main', __name__)

# Path to the trained model
MODEL_PATH = "perceptron_model.joblib"

def get_model():
    """Load the model once at startup"""
    if not hasattr(current_app, 'perceptron_model'):
        try:
            current_app.perceptron_model = load_model(MODEL_PATH, Perceptron)
        except FileNotFoundError:
            current_app.perceptron_model = None
            print("Warning: Model file not found. Please train the model first.")
    return current_app.perceptron_model

@main.route('/')
def index():
    return render_template('main/index.html')

@main.route('/profile')
@login_required
def profile():
    return render_template('main/profile.html')

@main.route('/profile/settings', methods=['GET', 'POST'])
@login_required
def profile_settings():
    form = ProfileUpdateForm(
        original_username=current_user.username,
        original_email=current_user.email
    )
    
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.email = form.email.data
        
        # Handle profile picture upload
        if form.profile_picture.data:
            file = form.profile_picture.data
            if file and file.filename:
                # Secure the filename
                filename = secure_filename(file.filename)
                # Add user ID to make filename unique
                name, ext = os.path.splitext(filename)
                filename = f"{current_user.id}_{name}{ext}"
                
                # Save the file
                filepath = os.path.join(current_app.root_path, 'static', 'uploads', filename)
                file.save(filepath)
                
                # Resize image to 200x200 for consistency
                try:
                    with Image.open(filepath) as img:
                        img = img.convert('RGB')
                        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                        img.save(filepath, 'JPEG', quality=85)
                except Exception as e:
                    flash(f'Error processing image: {str(e)}', 'error')
                    return redirect(url_for('main.profile_settings'))
                
                # Update database
                current_user.profile_picture = filename
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('main.profile'))
    
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    
    return render_template('main/profile_settings.html', form=form)

@main.route('/profile/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    form = PasswordChangeForm()
    
    if form.validate_on_submit():
        if current_user.check_password(form.current_password.data):
            current_user.set_password(form.new_password.data)
            db.session.commit()
            flash('Password changed successfully!', 'success')
            return redirect(url_for('main.profile'))
        else:
            flash('Current password is incorrect', 'error')
    
    return render_template('main/change_password.html', form=form)

@main.route('/my-predictions')
@login_required
def my_predictions():
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
        Prediction.created_at.desc()
    ).paginate(
        page=page, per_page=10, error_out=False
    )
    return render_template('main/my_predictions.html', predictions=predictions)

@main.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and file.filename.lower().endswith('.png'):
            # Load the trained model using model_utils
            from app.ai.model_utils import load_model
            from app.ai.perceptron import Perceptron
            
            try:
                model = load_model('perceptron_model.joblib', Perceptron)
                
                # Preprocess the image
                img = Image.open(file.stream).convert('L')  # Convert to grayscale
                img = img.resize((28, 28))  # Resize to 28x28
                img_array = np.array(img, dtype=np.float32).flatten() / 255.0  # Normalize to [0,1]
                
                # Ensure it's a 2D array for prediction (samples, features)
                img_array = img_array.reshape(1, -1)
                
                # Make prediction
                prediction = model.predict(img_array)[0]
                
                # Calculate confidence based on distance from decision boundary
                weights, bias = model.get_weights()
                decision_value = np.dot(img_array[0], weights) + bias
                confidence = min(abs(decision_value) / 10.0, 1.0)  # Normalize to 0-1
                
                # Store prediction in database
                new_prediction = Prediction(
                    filename=file.filename,
                    prediction='Circle' if prediction == 0 else 'Square',
                    confidence=confidence,
                    user_id=current_user.id
                )
                db.session.add(new_prediction)
                db.session.commit()
                
                result = 'Circle' if prediction == 0 else 'Square'
                flash(f'Prediction: {result} (Confidence: {confidence:.2f})', 'success')
                
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
        else:
            flash('Please upload a PNG file', 'error')
    
    return render_template('main/predict.html')

@main.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    form = FeedbackForm()
    
    if form.validate_on_submit():
        feedback = Feedback(
            rating=form.rating.data,
            comment=form.comment.data,
            is_public=form.is_public.data,
            user_id=current_user.id
        )
        db.session.add(feedback)
        db.session.commit()
        
        flash('Thank you for your feedback!', 'success')
        return redirect(url_for('main.feedback'))
    
    return render_template('main/feedback.html', form=form)

@main.route('/my-feedback')
@login_required
def my_feedback():
    page = request.args.get('page', 1, type=int)
    feedbacks = Feedback.query.filter_by(user_id=current_user.id).order_by(
        Feedback.created_at.desc()
    ).paginate(
        page=page, per_page=10, error_out=False
    )
    return render_template('main/my_feedback.html', feedbacks=feedbacks) 