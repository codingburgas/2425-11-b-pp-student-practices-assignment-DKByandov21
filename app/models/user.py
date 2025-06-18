from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from app import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    profile_picture = db.Column(db.String(255), nullable=True)  # Store filename of profile picture
    role = db.Column(db.String(20), default='user')  # 'user' or 'admin'
    active = db.Column(db.Boolean, default=True)  # For deactivation
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        return self.role == 'admin'
    
    def is_active(self):
        return self.active
    
    def get_prediction_count(self):
        return len(self.predictions)
    
    def get_circle_predictions(self):
        return len([p for p in self.predictions if p.prediction == 'Circle'])
    
    def get_square_predictions(self):
        return len([p for p in self.predictions if p.prediction == 'Square'])
    
    def get_profile_picture_url(self):
        """Return the URL for the user's profile picture or default"""
        if self.profile_picture:
            return f'/static/uploads/{self.profile_picture}'
        # Use a default avatar from a service like UI Avatars
        return f'https://ui-avatars.com/api/?name={self.username}&background=random&color=fff&size=200'
    
    def __repr__(self):
        return f'<User {self.username}>' 