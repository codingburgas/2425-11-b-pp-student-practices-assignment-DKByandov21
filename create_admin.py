from app import app, db
from models import User

def create_admin():
    with app.app_context():
        # Check if admin already exists
        admin = User.query.filter_by(role='admin').first()
        if admin:
            print(f"Admin already exists: {admin.username}")
            return
        
        # Create admin user
        admin = User(
            username='admin',
            email='admin@example.com',
            role='admin'
        )
        admin.set_password('admin123')
        
        db.session.add(admin)
        db.session.commit()
        
        print("Admin user created successfully!")
        print("Username: admin")
        print("Email: admin@example.com")
        print("Password: admin123")
        print("\nPlease change the password after first login!")

if __name__ == '__main__':
    create_admin() 