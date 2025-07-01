import joblib

def save_model(model, path):
    """
    Save the trained model (Perceptron or LogisticRegression) to a file using joblib.
    """
    # Save only the necessary attributes for reloading
    model_data = {
        'weights': model.weights,
        'bias': model.bias,
        'n_features': model.n_features,
        'n_iterations': model.n_iterations,
        'model_type': model.__class__.__name__
    }
    
    # Add loss history for LogisticRegression
    if hasattr(model, 'loss_history'):
        model_data['loss_history'] = model.loss_history
    
    joblib.dump(model_data, path)
    print(f"Model saved to {path}")

def load_model(path, model_class):
    """
    Load a model from a file and return an instance with restored weights and bias.
    """
    model_data = joblib.load(path)
    model = model_class()
    model.weights = model_data['weights']
    model.bias = model_data['bias']
    model.n_features = model_data['n_features']
    model.n_iterations = model_data.get('n_iterations', 0)
    
    # Restore loss history for LogisticRegression
    if hasattr(model, 'loss_history') and 'loss_history' in model_data:
        model.loss_history = model_data['loss_history']
    
    print(f"Model loaded from {path}")
    return model 