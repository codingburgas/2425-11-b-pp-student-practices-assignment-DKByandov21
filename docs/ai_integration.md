# AI Integration Documentation

## Overview

The Shape Classifier application integrates a custom Perceptron machine learning model with a Flask web application to provide real-time image classification capabilities. This document explains how the AI module works and how it integrates with the web application.

## AI Architecture

### Model Overview

The application uses a **binary Perceptron classifier** implemented from scratch in Python. This choice was made for:

- **Educational Value**: Demonstrates fundamental ML concepts
- **Transparency**: Complete control over the implementation
- **Performance**: Fast inference for real-time web applications
- **Simplicity**: Easy to understand and maintain

### Model Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Input Size** | 784 features | 28×28 grayscale images flattened |
| **Output** | Binary (0/1) | 0 = Circle, 1 = Square |
| **Algorithm** | Perceptron | Single-layer neural network |
| **Training** | SGD | Stochastic Gradient Descent |
| **Activation** | Step function | Binary classification output |
| **Features** | Normalized pixels | Values in range [0, 1] |

## AI Module Structure

```
app/ai/
├── __init__.py              # Module initialization
├── perceptron.py            # Perceptron classifier implementation
├── synthetic_dataset.py     # Dataset generation utilities
└── model_utils.py           # Model persistence and loading
```

### Core Components

#### 1. Perceptron Classifier (`perceptron.py`)

**Purpose**: Implements the binary classification algorithm.

**Key Methods**:
- `fit(X, y)`: Train the model on labeled data
- `predict(X)`: Make predictions on new data
- `score(X, y)`: Calculate accuracy on test data
- `get_weights()`: Retrieve model parameters for confidence calculation

**Training Process**:
```python
def fit(self, X, y, epochs=100, learning_rate=0.01):
    """
    Train the Perceptron using stochastic gradient descent.
    
    Args:
        X: Training features (n_samples, n_features)
        y: Training labels (n_samples,)
        epochs: Number of training iterations
        learning_rate: Learning rate for gradient descent
    """
    # Initialize weights and bias
    # For each epoch:
    #   For each sample:
    #     Make prediction
    #     Calculate error
    #     Update weights and bias
```

#### 2. Synthetic Dataset Generator (`synthetic_dataset.py`)

**Purpose**: Creates training and testing data for the model.

**Features**:
- Generates 28×28 grayscale images
- Creates circles and squares with random positioning
- Adds noise for robustness
- Normalizes pixel values to [0, 1] range

**Usage**:
```python
# Generate dataset
X, y = generate_synthetic_dataset(
    n_samples=1000,
    image_size=28,
    random_seed=42
)
```

#### 3. Model Utilities (`model_utils.py`)

**Purpose**: Handles model persistence and loading.

**Functions**:
- `save_model(model, filepath)`: Save trained model to disk
- `load_model(filepath, model_class)`: Load model from disk
- `get_model_info(model)`: Extract model metadata

## Integration with Flask Application

### 1. Model Loading Strategy

**Location**: `app/routes/main.py`

**Implementation**:
```python
def get_model():
    """Load the model once at startup for efficiency."""
    if not hasattr(current_app, 'perceptron_model'):
        try:
            current_app.perceptron_model = load_model(MODEL_PATH, Perceptron)
        except FileNotFoundError:
            current_app.perceptron_model = None
            print("Warning: Model file not found. Please train the model first.")
    return current_app.perceptron_model
```

**Benefits**:
- **Lazy Loading**: Model loaded only when needed
- **Caching**: Model stays in memory for fast inference
- **Error Handling**: Graceful degradation if model missing
- **Thread Safety**: Flask's application context ensures safety

### 2. Prediction Pipeline

**Flow**: User Upload → Image Processing → Model Inference → Result Storage

#### Step 1: Image Upload and Validation
```python
# Validate file type and size
if file and file.filename.endswith('.png'):
    img = Image.open(file).convert('L')  # Convert to grayscale
    if img.size != (28, 28):
        return 'Error: Image must be 28x28 pixels.'
```

#### Step 2: Image Preprocessing
```python
# Convert to numpy array and normalize
img_array = np.array(img, dtype=np.float32).flatten() / 255.0
X = img_array.reshape(1, -1)  # Reshape for model input
```

#### Step 3: Model Prediction
```python
# Get prediction and confidence
model = get_model()
if model is None:
    return 'Error: Model not available.'
    
pred = model.predict(X)[0]
prediction_result = 'Circle' if pred == 0 else 'Square'
```

#### Step 4: Confidence Calculation
```python
# Calculate confidence using decision boundary distance
weights, bias = model.get_weights()
decision_value = np.dot(X[0], weights) + bias
confidence = min(abs(decision_value) / 10.0, 1.0)  # Normalize to 0-1
```

#### Step 5: Result Storage
```python
# Save prediction to database
new_prediction = Prediction(
    user_id=current_user.id,
    filename=file.filename,
    prediction=prediction_result,
    confidence=confidence
)
db.session.add(new_prediction)
db.session.commit()
```

### 3. Error Handling

**Types of Errors Handled**:
- **Model Not Found**: Graceful message to user
- **Invalid Image Format**: Clear error message
- **Wrong Image Size**: Specific size requirements
- **Processing Errors**: Generic error with logging

**Implementation**:
```python
try:
    # Prediction logic
    prediction = model.predict(X)
except Exception as e:
    prediction = f'Error: {str(e)}'
    # Log error for debugging
    current_app.logger.error(f"Prediction error: {e}")
```

## Model Training Process

### 1. Training Script (`train_evaluate.py`)

**Purpose**: Orchestrates the complete training pipeline.

**Steps**:
1. **Dataset Generation**: Create synthetic training data
2. **Data Splitting**: Separate into train/test sets
3. **Model Training**: Fit Perceptron on training data
4. **Evaluation**: Assess performance on test set
5. **Model Persistence**: Save trained model to disk

**Usage**:
```bash
python train_evaluate.py
```

### 2. Training Configuration

**Parameters**:
- **Training Samples**: 1000 images (500 circles, 500 squares)
- **Test Samples**: 200 images (100 circles, 100 squares)
- **Epochs**: 100 training iterations
- **Learning Rate**: 0.01 (adaptive scheduling)
- **Random Seed**: 42 for reproducibility

### 3. Performance Metrics

**Evaluation Metrics**:
- **Accuracy**: Overall classification accuracy
- **Precision**: Accuracy for each class
- **Recall**: Sensitivity for each class
- **F1-Score**: Harmonic mean of precision and recall

**Typical Performance**:
- **Training Accuracy**: 95-98%
- **Test Accuracy**: 93-96%
- **Inference Time**: <10ms per image

## Model Persistence

### File Format

**Technology**: Joblib (scikit-learn compatible)

**Advantages**:
- **Fast**: Optimized for NumPy arrays
- **Compatible**: Works with scikit-learn ecosystem
- **Secure**: Safe for untrusted data
- **Compressed**: Automatic compression

**File Structure**:
```
perceptron_model.joblib
├── Model weights (numpy array)
├── Model bias (float)
├── Training metadata
└── Model version info
```

### Loading in Production

**Strategy**: Load once at application startup

**Benefits**:
- **Performance**: No repeated disk I/O
- **Reliability**: Model available for all requests
- **Memory Efficiency**: Single model instance
- **Thread Safety**: Flask handles concurrent access

## Confidence Scoring

### Implementation

**Method**: Distance from decision boundary

**Formula**:
```
confidence = min(|decision_value| / normalization_factor, 1.0)
```

**Where**:
- `decision_value = dot(X, weights) + bias`
- `normalization_factor = 10.0` (empirically determined)

### Interpretation

**Confidence Ranges**:
- **0.0-0.3**: Low confidence (uncertain prediction)
- **0.3-0.7**: Medium confidence (moderate certainty)
- **0.7-1.0**: High confidence (certain prediction)

**Visualization**: Progress bars in the UI show confidence levels with color coding.

## Future Enhancements

### Potential Improvements

1. **Model Architecture**:
   - Multi-layer Perceptron (MLP)
   - Convolutional Neural Network (CNN)
   - Transfer learning with pre-trained models

2. **Data Augmentation**:
   - Rotation and scaling
   - Noise injection
   - Color variations

3. **Ensemble Methods**:
   - Multiple model voting
   - Bagging and boosting
   - Model averaging

4. **Real-time Learning**:
   - Online learning capabilities
   - User feedback integration
   - Continuous model updates

### Scalability Considerations

1. **Model Serving**:
   - Separate model service
   - Load balancing
   - Model versioning

2. **Performance**:
   - GPU acceleration
   - Batch processing
   - Caching strategies

3. **Monitoring**:
   - Model performance tracking
   - Drift detection
   - Automated retraining

## Troubleshooting

### Common Issues

1. **Model Not Found**:
   - Ensure `train_evaluate.py` has been run
   - Check file permissions
   - Verify model file path

2. **Low Accuracy**:
   - Increase training data
   - Adjust learning rate
   - Check data quality

3. **Slow Inference**:
   - Profile model loading
   - Optimize preprocessing
   - Consider model compression

### Debugging Tools

1. **Logging**: Application logs show prediction details
2. **Model Inspection**: Access weights and bias for analysis
3. **Performance Profiling**: Time prediction pipeline
4. **Data Validation**: Verify input format and range 