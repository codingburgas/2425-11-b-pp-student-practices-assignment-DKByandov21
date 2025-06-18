# Shape Classifier - AI-Powered Image Classification Web Application

## Project Overview

Shape Classifier is a Flask-based web application that uses machine learning to classify 28x28 grayscale images as either circles or squares. The application features a custom Perceptron classifier, user authentication, role-based access control, and comprehensive admin functionality.

### Key Features

- **AI-Powered Classification**: Custom Perceptron model for binary image classification
- **User Authentication**: Secure registration, login, and session management
- **Role-Based Access Control**: User and admin roles with different permissions
- **Admin Dashboard**: Comprehensive user and prediction management
- **Prediction History**: Track and visualize user prediction history with confidence scores
- **Modern UI**: Responsive Bootstrap 5 interface with intuitive navigation
- **Database Integration**: SQLAlchemy ORM with PostgreSQL/SQLite support
- **Modular Architecture**: Clean blueprint-based Flask application structure

## Technology Stack

- **Backend**: Flask 2.3+, Python 3.11+
- **Database**: SQLAlchemy ORM, PostgreSQL/SQLite
- **Authentication**: Flask-Login, Flask-WTF
- **Frontend**: Bootstrap 5, Font Awesome, HTML5/CSS3
- **AI/ML**: NumPy, PIL (Pillow), Custom Perceptron Implementation
- **Model Persistence**: Joblib for model serialization

## Installation Instructions

### Prerequisites

- Python 3.11 or higher
- pip (Python package installer)
- Git

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd shape-classifier
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the project root:

```env
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///instance/app.db
```

### Step 5: Database Setup

```bash
# Initialize database migrations
python -m flask db init
python -m flask db migrate -m "Initial migration"
python -m flask db upgrade
```

### Step 6: Train the AI Model

```bash
# Generate synthetic dataset and train the model
python train_evaluate.py
```

### Step 7: Create Admin User

```bash
python create_admin.py
```

### Step 8: Run the Application

```bash
python run.py
```

The application will be available at `http://127.0.0.1:5000`

## Project Structure

```
shape-classifier/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── ai/                      # AI/ML modules
│   │   ├── __init__.py
│   │   ├── perceptron.py        # Custom Perceptron implementation
│   │   ├── synthetic_dataset.py # Dataset generation
│   │   └── model_utils.py       # Model persistence utilities
│   ├── models/                  # Database models
│   │   ├── __init__.py
│   │   ├── user.py             # User model
│   │   └── prediction.py       # Prediction model
│   ├── routes/                  # Application routes
│   │   ├── __init__.py
│   │   ├── auth.py             # Authentication routes
│   │   ├── main.py             # Main application routes
│   │   └── admin.py            # Admin routes
│   ├── templates/               # Jinja2 templates
│   │   ├── base.html           # Base template
│   │   ├── auth/               # Authentication templates
│   │   ├── main/               # Main application templates
│   │   ├── admin/              # Admin templates
│   │   └── components/         # Reusable components
│   ├── static/                  # Static files (CSS, JS, images)
│   ├── forms.py                # WTForms definitions
│   └── decorators.py           # Custom decorators
├── migrations/                  # Database migrations
├── instance/                    # Instance-specific files
├── config.py                   # Configuration settings
├── run.py                      # Application entry point
├── requirements.txt            # Python dependencies
├── train_evaluate.py           # Model training script
├── create_admin.py             # Admin user creation script
└── README.md                   # This file
```

## Usage Guide

### For Regular Users

1. **Registration**: Create an account with username, email, and password
2. **Login**: Access your personalized dashboard
3. **Upload Images**: Submit 28x28 grayscale PNG images for classification
4. **View Results**: See predictions with confidence scores
5. **History**: Review your prediction history and statistics

### For Administrators

1. **Dashboard**: View comprehensive system statistics
2. **User Management**: Manage user accounts, roles, and status
3. **Prediction Monitoring**: Track all system predictions
4. **Statistics**: Analyze system performance and user activity

## API Endpoints

### Authentication
- `GET/POST /register` - User registration
- `GET/POST /login` - User login
- `GET /logout` - User logout

### Main Application
- `GET /` - Home page
- `GET /profile` - User profile
- `GET /predictions` - User prediction history
- `GET/POST /predict` - Image prediction

### Admin (requires admin role)
- `GET /admin/` - Admin dashboard
- `GET /admin/users` - User management
- `GET /admin/predictions` - All predictions
- `GET /admin/statistics` - System statistics
- `POST /admin/user/<id>/promote` - Promote user to admin
- `POST /admin/user/<id>/demote` - Demote admin to user
- `POST /admin/user/<id>/activate` - Activate user
- `POST /admin/user/<id>/deactivate` - Deactivate user
- `POST /admin/user/<id>/delete` - Delete user

## AI Model Details

### Perceptron Classifier

The application uses a custom binary Perceptron classifier implemented from scratch:

- **Input**: 28x28 grayscale images (784 features)
- **Output**: Binary classification (0 = Circle, 1 = Square)
- **Training**: Stochastic gradient descent with learning rate scheduling
- **Features**: Normalized pixel values (0-1 range)
- **Performance**: Typically achieves 95%+ accuracy on synthetic data

### Model Training Process

1. **Dataset Generation**: Creates synthetic 28x28 images with circles and squares
2. **Data Preprocessing**: Normalizes pixel values and flattens images
3. **Training**: Uses gradient descent to optimize weights and bias
4. **Validation**: Evaluates model performance on test set
5. **Persistence**: Saves trained model using Joblib

### Integration with Flask

- **Model Loading**: Loaded once at application startup
- **Prediction Pipeline**: Image upload → preprocessing → prediction → confidence calculation
- **Result Storage**: Predictions stored in database with metadata
- **Error Handling**: Graceful handling of invalid images and model errors

## Database Schema

The application uses two main models:

### User Model
- `id`: Primary key
- `username`: Unique username
- `email`: Unique email address
- `password_hash`: Hashed password
- `role`: User role (user/admin)
- `active`: Account status
- `created_at`: Registration timestamp

### Prediction Model
- `id`: Primary key
- `user_id`: Foreign key to User
- `filename`: Original filename
- `prediction`: Classification result
- `confidence`: Prediction confidence score
- `created_at`: Prediction timestamp

**Relationships**: One-to-many (User → Predictions)

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code formatting
- Use PEP 257 for docstring formatting
- Implement proper error handling
- Write meaningful commit messages

### Testing
- Test all routes and functionality
- Validate form inputs and outputs
- Test admin permissions and restrictions
- Verify AI model integration

### Security
- Password hashing with Werkzeug
- CSRF protection on forms
- Role-based access control
- Input validation and sanitization

## Deployment

### Production Considerations

1. **Environment Variables**: Set proper SECRET_KEY and DATABASE_URL
2. **Database**: Use PostgreSQL for production
3. **Static Files**: Configure proper static file serving
4. **Logging**: Implement application logging
5. **Security**: Enable HTTPS and security headers
6. **Performance**: Use production WSGI server (Gunicorn)

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "run.py"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please open an issue in the repository or contact the development team.

## Acknowledgments

- Flask community for the excellent web framework
- Bootstrap team for the responsive UI components
- NumPy and PIL teams for the scientific computing tools 