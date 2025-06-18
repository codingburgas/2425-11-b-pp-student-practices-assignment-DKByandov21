from datetime import datetime
from app import db

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rating = db.Column(db.Integer, nullable=False)  # 1-5 stars
    comment = db.Column(db.Text, nullable=True)  # Optional comment
    is_public = db.Column(db.Boolean, default=True)  # Visibility flag
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to user
    user = db.relationship('User', backref='feedbacks')
    
    def __repr__(self):
        return f'<Feedback {self.id} by User {self.user_id}>'
    
    @property
    def rating_stars(self):
        """Return rating as star emoji string"""
        return '⭐' * self.rating
    
    @property
    def rating_text(self):
        """Return rating as text description"""
        ratings = {
            1: 'Very Poor',
            2: 'Poor', 
            3: 'Average',
            4: 'Good',
            5: 'Excellent'
        }
        return ratings.get(self.rating, 'Unknown') 