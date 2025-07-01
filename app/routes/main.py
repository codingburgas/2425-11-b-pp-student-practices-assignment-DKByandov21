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
from app.ai import Perceptron, load_model
from app.forms import ProfileUpdateForm, PasswordChangeForm, FeedbackForm
import io
from datetime import datetime

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

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
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        # Get selected model type
        model_type = request.form.get('model_type', 'perceptron')

        if file and allowed_file(file.filename):
            try:
                # Read and process the image
                image = Image.open(io.BytesIO(file.read()))

                # Convert to grayscale and resize to 28x28
                image = image.convert('L')
                image = image.resize((28, 28))

                # Convert to numpy array and normalize
                image_array = np.array(image, dtype=np.float32)
                image_array = image_array / 255.0  # Normalize to 0-1
                image_array = image_array.flatten()  # Flatten to 1D array

                # Load the appropriate model
                if model_type == 'logistic':
                    from app.ai import LogisticRegression
                    model = load_model('logistic_model.joblib', LogisticRegression)
                    # Make prediction with probability
                    prediction = model.predict(image_array.reshape(1, -1))[0]
                    probability = model.predict_proba(image_array.reshape(1, -1))[0][prediction]  # Probability of the predicted class
                    confidence = probability #if prediction == 1 else (1 - probability) # Use probability as confidence
                    model_name = "Logistic Regression"
                else:
                    model = load_model('perceptron_model.joblib', Perceptron)
                    # Make prediction
                    prediction = model.predict(image_array.reshape(1, -1))[0]
                    # Calculate confidence (distance from decision boundary)
                    weights, bias = model.get_weights()
                    z = np.dot(image_array, weights) + bias
                    confidence = abs(z)  # Distance from decision boundary
                    confidence = min(confidence * 10, 1.0)  # Scale and cap at 1.0
                    model_name = "Perceptron"

                # Convert prediction to label
                predicted_class = 'Square' if prediction == 1 else 'Circle'

                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{current_user.id}_{timestamp}_{filename}"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

                # Save the processed image
                processed_image = Image.fromarray((image_array.reshape(28, 28) * 255).astype(np.uint8))
                processed_image.save(filepath)

                # Save prediction to database
                new_prediction = Prediction(
                    user_id=current_user.id,
                    image_filename=filename,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    timestamp=datetime.now()
                )
                db.session.add(new_prediction)
                db.session.commit()

                flash(f'Prediction complete! Model: {model_name}, Result: {predicted_class} (Confidence: {confidence:.2f})', 'success')
                return render_template('main/predict.html', 
                                     prediction=predicted_class, 
                                     confidence=confidence,
                                     image_filename=filename,
                                     model_type=model_type,
                                     model_name=model_name)

            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'danger')

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