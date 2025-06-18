# 🧠 Binary Image Classifier – AI Web App for Shape Recognition


<img src="./docs/banner.png" alt="Banner" />

## 📌 Project Overview

Shape Classifier is a Flask-based web application that uses machine learning to classify 28×28 grayscale images as either **circles** or **squares**.

## ✨ Key Features

- 🤖 **AI-Powered Classification** – Custom Perceptron model
- 🔐 **User Authentication** – Registration, login, and session management
- 🧑‍⚖️ **Role-Based Access Control** – Users and administrators
- 📊 **Prediction History** – View results with confidence scores
- 🧑‍💻 **Modern UI** – Bootstrap 5 responsive interface
- 💾 **Database Integration** – SQLAlchemy + PostgreSQL/SQLite
- 🧱 **Modular Architecture** – Flask Blueprints

## 🧰 Technology Stack

- 🐍 Backend: **Flask 2.3+, Python 3.11+**
- 🗄️ Database: **PostgreSQL/SQLite + SQLAlchemy**
- 🔐 Auth: **Flask-Login, Flask-WTF**
- 🎨 Frontend: **Bootstrap 5, HTML5, CSS3**
- 🧠 AI/ML: **NumPy, Pillow, Custom Perceptron**
- 💾 Persistence: **Joblib**

## 🛠️ Installation Instructions

### 📦 Prerequisites

- Python 3.11+
- pip
- Git

### 🚀 Steps

```bash
git clone <repository-url>
cd shape-classifier

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### ⚙️ Environment Configuration

Create a `.env` file:

```env
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///instance/app.db
```

### 🧠 Train the AI Model

```bash
python train_evaluate.py
```

### 👑 Create Admin User

```bash
python create_admin.py
```

### ▶️ Run the Application

```bash
python run.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🗂️ Project Structure

```plaintext
shape-classifier/
├── app/
│   ├── ai/                # AI logic
│   ├── models/            # Database models
│   ├── routes/            # Flask routes
│   ├── templates/         # HTML templates
│   └── static/            # CSS/JS/images
├── config.py
├── run.py
├── requirements.txt
├── train_evaluate.py
└── create_admin.py
```

---

## 👤 User Guide

### 👥 Regular Users

1. Register and log in
2. Upload 28x28 grayscale images
3. View predictions and confidence
4. Analyze prediction history

### 🛡️ Admins

1. Access system dashboard
2. Manage users and roles
3. Monitor predictions
4. View usage statistics

---

## 🔌 API Endpoints

### 🧾 Authentication

- `GET/POST /register`
- `GET/POST /login`
- `GET /logout`

### 📈 Main App

- `GET /`
- `GET /profile`
- `GET /predictions`
- `POST /predict`

### 🔧 Admin (requires role)

- `GET /admin/`
- `GET /admin/users`
- `GET /admin/predictions`
- `GET /admin/statistics`
- `POST /admin/user/<id>/...` – Promote, deactivate, etc.

---

## 🧠 AI Model Details

- Input: **784 pixels (28×28)**
- Output: **0 = Circle, 1 = Square**
- Training: **SGD**
- Performance: **95%+ accuracy**

---


## 🗃️ Database Schema

### 👤 User

- `id`: Primary key
- `username`: Unique username
- `email`: Unique email
- `password_hash`: Hashed password
- `created_at`: Registration timestamp
- `role`: User role (user/admin)
- `active`: Account status
- `profile_picture`: Profile image path

### 📷 Prediction

- `id`: Primary key
- `user_id`: Foreign key to User
- `filename`: Original image name
- `prediction`: Classification result
- `confidence`: Confidence score
- `created_at`: Timestamp

### 💬 Feedback

- `id`: Primary key
- `rating`: Integer score (e.g., 1–5)
- `comment`: Optional user comment
- `is_public`: Boolean visibility flag
- `user_id`: Foreign key to User
- `created_at`: Timestamp

➡️ Relationships:
- User → Predictions (1:N)
- User → Feedback (1:N)


## 🧪 Development Guidelines

- Follow PEP 8 & PEP 257
- Validate input/output
- Handle errors properly
- Write meaningful commits
- Write tests for key functionality

---

## 🚀 Deployment

### 🔒 Security

- Set secure `SECRET_KEY`, `DATABASE_URL`
- Use HTTPS, enable CSRF protection
- Enable logs and headers

### 🐳 Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "run.py"]
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch
3. Make changes
4. Add tests (if needed)
5. Submit a PR

## 📄 License

MIT License

## 🙏 Acknowledgments

- Flask
- Bootstrap
- NumPy & Pillow