from flask import Blueprint, render_template, redirect, url_for, flash, jsonify, request
from flask_login import login_required, current_user
from app import db
from app.models.user import User
from app.models.prediction import Prediction
from app.models.feedback import Feedback
from app.decorators import admin_required
import re
from collections import Counter
from sqlalchemy import or_, and_

admin = Blueprint('admin', __name__, url_prefix='/admin')

@admin.route('/')
@login_required
@admin_required
def dashboard():
    # Get comprehensive statistics for admin dashboard
    total_users = User.query.count()
    active_users = User.query.filter_by(active=True).count()
    total_predictions = Prediction.query.count()
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    users = User.query.all()
    
    # Aggregated statistics
    circle_predictions = Prediction.query.filter_by(prediction='Circle').count()
    square_predictions = Prediction.query.filter_by(prediction='Square').count()
    
    # Calculate average confidence
    predictions_with_confidence = Prediction.query.filter(Prediction.confidence.isnot(None)).all()
    avg_confidence = 0
    if predictions_with_confidence:
        avg_confidence = sum(p.confidence for p in predictions_with_confidence) / len(predictions_with_confidence)
    
    return render_template('admin/admin_dashboard.html', 
                         total_users=total_users,
                         active_users=active_users,
                         total_predictions=total_predictions,
                         recent_predictions=recent_predictions,
                         users=users,
                         circle_predictions=circle_predictions,
                         square_predictions=square_predictions,
                         avg_confidence=avg_confidence)

@admin.route('/users')
@login_required
@admin_required
def users():
    users = User.query.all()
    return render_template('admin/admin_users.html', users=users)

@admin.route('/predictions')
@login_required
@admin_required
def predictions():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template('admin/admin_predictions.html', predictions=predictions)

@admin.route('/statistics')
@login_required
@admin_required
def statistics():
    # Comprehensive statistics
    total_users = User.query.count()
    active_users = User.query.filter_by(active=True).count()
    total_predictions = Prediction.query.count()
    
    # Prediction breakdown
    circle_predictions = Prediction.query.filter_by(prediction='Circle').count()
    square_predictions = Prediction.query.filter_by(prediction='Square').count()
    
    # User activity statistics
    users_with_predictions = User.query.join(Prediction).distinct().count()
    most_active_user = db.session.query(User, db.func.count(Prediction.id).label('pred_count'))\
        .join(Prediction)\
        .group_by(User.id)\
        .order_by(db.func.count(Prediction.id).desc())\
        .first()
    
    # Average predictions per user
    avg_predictions_per_user = total_predictions / total_users if total_users > 0 else 0
    
    # Confidence statistics
    predictions_with_confidence = Prediction.query.filter(Prediction.confidence.isnot(None)).all()
    avg_confidence = 0
    if predictions_with_confidence:
        avg_confidence = sum(p.confidence for p in predictions_with_confidence) / len(predictions_with_confidence)
    
    return render_template('admin/admin_statistics.html',
                         total_users=total_users,
                         active_users=active_users,
                         total_predictions=total_predictions,
                         circle_predictions=circle_predictions,
                         square_predictions=square_predictions,
                         users_with_predictions=users_with_predictions,
                         most_active_user=most_active_user,
                         avg_predictions_per_user=avg_predictions_per_user,
                         avg_confidence=avg_confidence)

@admin.route('/user/<int:user_id>/promote')
@login_required
@admin_required
def promote_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.role == 'user':
        user.role = 'admin'
        db.session.commit()
        flash(f'{user.username} has been promoted to admin.', 'success')
    else:
        flash(f'{user.username} is already an admin.', 'info')
    return redirect(url_for('admin.users'))

@admin.route('/user/<int:user_id>/demote')
@login_required
@admin_required
def demote_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot demote yourself.', 'error')
    elif user.role == 'admin':
        user.role = 'user'
        db.session.commit()
        flash(f'{user.username} has been demoted to user.', 'success')
    else:
        flash(f'{user.username} is already a user.', 'info')
    return redirect(url_for('admin.users'))

@admin.route('/user/<int:user_id>/deactivate')
@login_required
@admin_required
def deactivate_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot deactivate yourself.', 'error')
    elif user.active:
        user.active = False
        db.session.commit()
        flash(f'{user.username} has been deactivated.', 'success')
    else:
        flash(f'{user.username} is already deactivated.', 'info')
    return redirect(url_for('admin.users'))

@admin.route('/user/<int:user_id>/activate')
@login_required
@admin_required
def activate_user(user_id):
    user = User.query.get_or_404(user_id)
    if not user.active:
        user.active = True
        db.session.commit()
        flash(f'{user.username} has been activated.', 'success')
    else:
        flash(f'{user.username} is already active.', 'info')
    return redirect(url_for('admin.users'))

@admin.route('/user/<int:user_id>/delete')
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot delete yourself.', 'error')
    else:
        username = user.username
        # Delete all predictions by this user first
        Prediction.query.filter_by(user_id=user.id).delete()
        # Delete all feedback by this user
        Feedback.query.filter_by(user_id=user.id).delete()
        # Delete the user
        db.session.delete(user)
        db.session.commit()
        flash(f'{username} and all their predictions and feedback have been deleted.', 'success')
    return redirect(url_for('admin.users'))

@admin.route('/feedback')
@login_required
@admin_required
def feedback():
    page = request.args.get('page', 1, type=int)
    rating_filter = request.args.get('rating', type=int)
    visibility_filter = request.args.get('visibility')
    search = request.args.get('search', '').strip()

    query = Feedback.query
    if rating_filter:
        query = query.filter(Feedback.rating == rating_filter)
    if visibility_filter in ['public', 'private']:
        query = query.filter(Feedback.is_public == (visibility_filter == 'public'))
    if search:
        query = query.filter(or_(Feedback.comment.ilike(f'%{search}%'), User.username.ilike(f'%{search}%')))
    query = query.order_by(Feedback.created_at.desc())

    feedbacks = query.paginate(page=page, per_page=15, error_out=False)

    # Analytics
    total_feedback = Feedback.query.count()
    rating_dist = [Feedback.query.filter_by(rating=i).count() for i in range(1, 6)]
    public_count = Feedback.query.filter_by(is_public=True).count()
    private_count = Feedback.query.filter_by(is_public=False).count()

    return render_template(
        'admin/feedback.html',
        feedbacks=feedbacks,
        total_feedback=total_feedback,
        rating_dist=rating_dist,
        public_count=public_count,
        private_count=private_count,
        rating_filter=rating_filter,
        visibility_filter=visibility_filter,
        search=search
    )

@admin.route('/feedback/<int:feedback_id>/delete')
@login_required
@admin_required
def delete_feedback(feedback_id):
    feedback = Feedback.query.get_or_404(feedback_id)
    db.session.delete(feedback)
    db.session.commit()
    flash('Feedback deleted successfully.', 'success')
    return redirect(url_for('admin.feedback'))

@admin.route('/feedback/stats')
@login_required
@admin_required
def feedback_stats():
    """API endpoint for feedback statistics (for charts)"""
    # Rating distribution
    rating_dist = {}
    for i in range(1, 6):
        rating_dist[f'{i} Star'] = Feedback.query.filter_by(rating=i).count()
    
    # Visibility distribution
    visibility_dist = {
        'Public': Feedback.query.filter_by(is_public=True).count(),
        'Private': Feedback.query.filter_by(is_public=False).count()
    }
    
    return jsonify({
        'rating_distribution': rating_dist,
        'visibility_distribution': visibility_dist
    }) 