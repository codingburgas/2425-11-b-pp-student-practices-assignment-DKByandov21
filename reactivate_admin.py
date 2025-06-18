from app import create_app, db
from app.models.user import User

app = create_app()

if __name__ == "__main__":
    email = input("Enter the admin email to reactivate: ").strip()
    with app.app_context():
        admin = User.query.filter_by(email=email).first()
        if admin:
            admin.active = True
            db.session.commit()
            print(f"Admin '{admin.username}' reactivated!")
        else:
            print("Admin not found.") 