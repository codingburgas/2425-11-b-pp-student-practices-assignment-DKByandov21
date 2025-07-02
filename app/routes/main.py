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

# Път до обучен модел
MODEL_PATH = "perceptron_model.joblib"

def get_model():
    """Зареждане на модела веднъж при стартиране"""
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
    # Изчисляване на точността и метриките на потребителя
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    user_feedbacks = Feedback.query.filter_by(user_id=current_user.id).all()
    
    # Изчисляване на основни статистики
    total_predictions = len(user_predictions)
    total_feedback = len(user_feedbacks)
    
    # Изчисляване на средна увереност
    avg_confidence = 0
    if total_predictions > 0:
        avg_confidence = sum(p.confidence for p in user_predictions) / total_predictions
    
    # Изчисляване на средна оценка
    avg_rating = 0
    if total_feedback > 0:
        avg_rating = sum(f.rating for f in user_feedbacks) / total_feedback
    
    # Обобщени данни за предсказанията
    prediction_stats = {
        'total': total_predictions,
        'avg_confidence': avg_confidence,
        'avg_rating': avg_rating,
        'total_feedback': total_feedback
    }
    
    return render_template('main/profile.html', prediction_stats=prediction_stats)

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

        # Обработка на качване на профилна снимка
        if form.profile_picture.data:
            file = form.profile_picture.data
            if file and file.filename:
                filename = secure_filename(file.filename)
                name, ext = os.path.splitext(filename)
                filename = f"{current_user.id}_{name}{ext}"

                filepath = os.path.join(current_app.root_path, 'static', 'uploads', filename)
                file.save(filepath)

                # Преоразмеряване на изображението до 200x200
                try:
                    with Image.open(filepath) as img:
                        img = img.convert('RGB')
                        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                        img.save(filepath, 'JPEG', quality=85)
                except Exception as e:
                    flash(f'Error processing image: {str(e)}', 'error')
                    return redirect(url_for('main.profile_settings'))

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
        # Проверка дали има избран файл
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        # Получаване на избрания модел
        model_type = request.form.get('model_type', 'perceptron')

        if file and allowed_file(file.filename):
            try:
                # Обработка на изображението
                image = Image.open(io.BytesIO(file.read()))
                image = image.convert('L')
                image = image.resize((28, 28))

                image_array = np.array(image, dtype=np.float32)
                image_array = image_array / 255.0
                image_array = image_array.flatten()

                if model_type == 'logistic':
                    from app.ai import LogisticRegression
                    model = load_model('app/ai/save_models/logistic_model.joblib', LogisticRegression)
                    prediction = model.predict(image_array.reshape(1, -1))[0]
                    probability = model.predict_proba(image_array.reshape(1, -1))[0]
                    confidence = probability if prediction == 1 else 1 - probability
                    model_name = "Logistic Regression"

                    from app.ai.metrics import ModelMetrics
                    log_loss_value = ModelMetrics.log_loss(np.array([prediction]), np.array([probability]))
                else:
                    model = load_model('app/ai/save_models/perceptron_model.joblib', Perceptron)
                    prediction = model.predict(image_array.reshape(1, -1))[0]
                    weights, bias = model.get_weights()
                    z = np.dot(image_array, weights) + bias
                    confidence = abs(z)
                    confidence = min(confidence * 10, 1.0)
                    model_name = "Perceptron"

                predicted_class = 'Square' if prediction == 1 else 'Circle'

                # Запазване на изображението
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{current_user.id}_{timestamp}_{filename}"
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

                processed_image = Image.fromarray((image_array.reshape(28, 28) * 255).astype(np.uint8))
                processed_image.save(filepath)

                # Запазване в базата
                new_prediction = Prediction(
                    user_id=current_user.id,
                    filename=filename,
                    prediction=predicted_class,
                    confidence=confidence
                )
                db.session.add(new_prediction)
                db.session.commit()

                flash(f'Prediction complete! Model: {model_name}, Result: {predicted_class} (Confidence: {confidence:.2f})', 'success')

                additional_metrics = {}
                if model_type == 'logistic':
                    additional_metrics['log_loss'] = log_loss_value
                    additional_metrics['probability'] = probability
                else:
                    additional_metrics['decision_distance'] = abs(z)

                return render_template('main/predict.html', 
                                     prediction=predicted_class, 
                                     confidence=confidence,
                                     image_filename=filename,
                                     model_type=model_type,
                                     model_name=model_name,
                                     additional_metrics=additional_metrics)

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
        try:
            rating_value = int(form.rating.data)
            if rating_value < 1 or rating_value > 5:
                flash('Invalid rating value. Please select a rating between 1 and 5 stars.', 'danger')
                return render_template('main/feedback.html', form=form)

            feedback = Feedback(
                rating=rating_value,
                comment=form.comment.data.strip() if form.comment.data else None,
                is_public=form.is_public.data,
                user_id=current_user.id
            )
            
            db.session.add(feedback)
            db.session.commit()

            flash(f'Thank you for your {feedback.rating_text.lower()} feedback! Your {rating_value}-star rating has been saved.', 'success')
            return redirect(url_for('main.feedback'))

        except (ValueError, TypeError) as e:
            flash('Error processing your rating. Please try again.', 'danger')
            current_app.logger.error(f"Feedback rating error: {e}")
        except Exception as e:
            db.session.rollback()
            flash('An error occurred while saving your feedback. Please try again.', 'danger')
            current_app.logger.error(f"Feedback save error: {e}")
    
    elif request.method == 'POST':
        if not form.rating.data:
            flash('Please select a star rating before submitting your feedback.', 'warning')
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{field.title()}: {error}', 'danger')

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

@main.route('/model-evaluation')
@login_required
def model_evaluation():
    """Показване на метрики и сравнение между модели."""
    try:
        from app.ai.metrics import ModelMetrics
        from app.ai.synthetic_dataset import create_synthetic_dataset
        from sklearn.metrics import confusion_matrix
        
        perceptron_model = load_model('app/ai/save_models/perceptron_model.joblib', Perceptron)
        
        try:
            from app.ai import LogisticRegression
            logistic_model = load_model('app/ai/save_models/logistic_model.joblib', LogisticRegression)
        except (FileNotFoundError, Exception) as e:
            print(f"Error loading logistic model: {str(e)}")
            logistic_model = None
        
        X_test, y_test = create_synthetic_dataset(n_samples_per_class=500, random_seed=42)
        
        perceptron_pred = perceptron_model.predict(X_test)
        perceptron_metrics = ModelMetrics.calculate_all_metrics(y_test, perceptron_pred)
        
        cm_perceptron = confusion_matrix(y_test, perceptron_pred)
        perceptron_cm = {
            'true_negatives': int(cm_perceptron[0, 0]),
            'false_positives': int(cm_perceptron[0, 1]),
            'false_negatives': int(cm_perceptron[1, 0]),
            'true_positives': int(cm_perceptron[1, 1]),
            'total': int(len(y_test))
        }
        
        logistic_metrics = None
        logistic_cm = None
        if logistic_model:
            logistic_pred = logistic_model.predict(X_test)
            logistic_metrics = ModelMetrics.calculate_all_metrics(y_test, logistic_pred)
            
            cm_logistic = confusion_matrix(y_test, logistic_pred)
            logistic_cm = {
                'true_negatives': int(cm_logistic[0, 0]),
                'false_positives': int(cm_logistic[0, 1]),
                'false_negatives': int(cm_logistic[1, 0]),
                'true_positives': int(cm_logistic[1, 1]),
                'total': int(len(y_test))
            }

        feature_importance = {
            'top_features': []
        }

        for i in range(20):
            feature_importance['top_features'].append({
                'feature_name': f'Pixel_{i*39}',
                'feature_idx': i*39,
                'information_gain': 0.1 - (i * 0.005)
            })
        
        return render_template('main/model_evaluation.html', 
                             perceptron_metrics=perceptron_metrics,
                             logistic_metrics=logistic_metrics,
                             perceptron_cm=perceptron_cm,
                             logistic_cm=logistic_cm,
                             feature_importance=feature_importance)
    except Exception as e:
        flash(f'Error loading model evaluation: {str(e)}', 'error')
        return redirect(url_for('main.predict'))
