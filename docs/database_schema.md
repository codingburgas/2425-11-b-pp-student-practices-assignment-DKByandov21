# Database Schema Documentation

## Overview

The Shape Classifier application uses a relational database with two main models: `User` and `Prediction`. The database is managed through SQLAlchemy ORM and supports both SQLite (development) and PostgreSQL (production).

## Entity Relationship Diagram

```
┌─────────────────┐         ┌──────────────────┐
│      User       │         │    Prediction    │
├─────────────────┤         ├──────────────────┤
│ id (PK)         │◄────────┤ id (PK)          │
│ username        │         │ user_id (FK)     │
│ email           │         │ filename         │
│ password_hash   │         │ prediction       │
│ role            │         │ confidence       │
│ active          │         │ created_at       │
│ created_at      │         └──────────────────┘
└─────────────────┘
```

## Model Definitions

### User Model

**Purpose**: Stores user account information and authentication data.

**Table**: `user`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | Integer | Primary Key, Auto-increment | Unique user identifier |
| `username` | String(80) | Unique, Not Null | User's display name |
| `email` | String(120) | Unique, Not Null | User's email address |
| `password_hash` | String(128) | Not Null | Hashed password using Werkzeug |
| `role` | String(20) | Default: 'user' | User role: 'user' or 'admin' |
| `active` | Boolean | Default: True | Account activation status |
| `created_at` | DateTime | Default: UTC now | Account creation timestamp |

**Relationships**:
- One-to-Many with Prediction (one user can have many predictions)

**Methods**:
- `set_password(password)`: Securely hash and store password
- `check_password(password)`: Verify password against hash
- `is_admin()`: Check if user has admin role
- `is_active()`: Check if account is active
- `get_prediction_count()`: Count user's predictions
- `get_circle_predictions()`: Count circle predictions
- `get_square_predictions()`: Count square predictions

### Prediction Model

**Purpose**: Stores image classification results and metadata.

**Table**: `prediction`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | Integer | Primary Key, Auto-increment | Unique prediction identifier |
| `user_id` | Integer | Foreign Key, Not Null | Reference to User.id |
| `filename` | String(255) | Not Null | Original uploaded filename |
| `prediction` | String(50) | Not Null | Classification result: 'Circle' or 'Square' |
| `confidence` | Float | Nullable | Prediction confidence score (0.0-1.0) |
| `created_at` | DateTime | Default: UTC now | Prediction timestamp |

**Relationships**:
- Many-to-One with User (many predictions belong to one user)

**Indexes**:
- `user_id`: For efficient user-based queries
- `created_at`: For chronological sorting
- `prediction`: For classification-based filtering

## Database Migrations

The application uses Flask-Migrate for database version control:

```bash
# Initialize migrations
flask db init

# Create migration
flask db migrate -m "Description of changes"

# Apply migration
flask db upgrade

# Rollback migration
flask db downgrade
```

## Sample Queries

### User Statistics
```sql
-- Count total users
SELECT COUNT(*) FROM user;

-- Count active users
SELECT COUNT(*) FROM user WHERE active = TRUE;

-- Count admin users
SELECT COUNT(*) FROM user WHERE role = 'admin';

-- Users with predictions
SELECT COUNT(DISTINCT user_id) FROM prediction;
```

### Prediction Statistics
```sql
-- Total predictions
SELECT COUNT(*) FROM prediction;

-- Predictions by type
SELECT prediction, COUNT(*) 
FROM prediction 
GROUP BY prediction;

-- Average confidence
SELECT AVG(confidence) FROM prediction WHERE confidence IS NOT NULL;

-- Recent predictions (last 7 days)
SELECT * FROM prediction 
WHERE created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;
```

### User Activity
```sql
-- Most active users
SELECT u.username, COUNT(p.id) as prediction_count
FROM user u
JOIN prediction p ON u.id = p.user_id
GROUP BY u.id, u.username
ORDER BY prediction_count DESC;

-- User prediction history
SELECT p.filename, p.prediction, p.confidence, p.created_at
FROM prediction p
WHERE p.user_id = ?
ORDER BY p.created_at DESC;
```

## Data Integrity

### Constraints
- **Unique Constraints**: Username and email must be unique
- **Foreign Key**: Prediction.user_id references User.id
- **Check Constraints**: Role must be 'user' or 'admin'
- **Not Null**: Essential fields cannot be null

### Validation
- Email format validation using WTForms
- Password strength requirements
- File type validation for uploads
- Confidence score range validation (0.0-1.0)

## Performance Considerations

### Indexing Strategy
- Primary keys are automatically indexed
- Foreign keys should be indexed for join performance
- Frequently queried columns (created_at, prediction) are indexed
- Composite indexes for complex queries

### Query Optimization
- Use eager loading for related data
- Implement pagination for large result sets
- Cache frequently accessed statistics
- Use database views for complex aggregations

## Backup and Recovery

### Backup Strategy
```bash
# SQLite backup
cp instance/app.db instance/app.db.backup

# PostgreSQL backup
pg_dump -U username -d database_name > backup.sql
```

### Recovery Process
```bash
# SQLite restore
cp instance/app.db.backup instance/app.db

# PostgreSQL restore
psql -U username -d database_name < backup.sql
```

## Security Considerations

### Data Protection
- Passwords are hashed using Werkzeug's security functions
- Sensitive data is not logged
- Database connections use parameterized queries
- Input validation prevents SQL injection

### Access Control
- Role-based permissions at application level
- Database user has minimal required privileges
- Regular security audits and updates
- Encrypted connections in production 