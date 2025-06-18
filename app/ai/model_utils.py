import joblib

def save_model(model, path):
    """
    Save the trained Perceptron model (weights, bias, and n_features) to a file using joblib.
    """
    # Save only the necessary attributes for reloading
    model_data = {
        'weights': model.weights,
        'bias': model.bias,
        'n_features': model.n_features,
        'n_iterations': model.n_iterations
    }
    joblib.dump(model_data, path)
    print(f"Model saved to {path}")

def load_model(path, perceptron_class):
    """
    Load a Perceptron model from a file and return an instance with restored weights and bias.
    """
    model_data = joblib.load(path)
    model = perceptron_class()
    model.weights = model_data['weights']
    model.bias = model_data['bias']
    model.n_features = model_data['n_features']
    model.n_iterations = model_data.get('n_iterations', 0)
    print(f"Model loaded from {path}")
    return model 