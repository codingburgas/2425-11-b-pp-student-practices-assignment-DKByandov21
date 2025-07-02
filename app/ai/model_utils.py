import joblib

def save_model(model, path):
    """
    Запазва обучен модел (Perceptron или LogisticRegression) във файл чрез joblib.
    """
    # Запазва само необходимите атрибути за повторно зареждане
    model_data = {
        'weights': model.weights,
        'bias': model.bias,
        'n_features': model.n_features,
        'n_iterations': model.n_iterations,
        'model_type': model.__class__.__name__
    }
    
    # Добавя историята на загубите, ако е LogisticRegression
    if hasattr(model, 'loss_history'):
        model_data['loss_history'] = model.loss_history
    
    joblib.dump(model_data, path)
    print(f"Моделът е запазен в {path}")

def load_model(path, model_class):
    """
    Зарежда модел от файл и връща инстанция с възстановени тегла и bias.
    
    """
    model_data = joblib.load(path)
    model = model_class()
    model.weights = model_data['weights']
    model.bias = model_data['bias']
    model.n_features = model_data['n_features']
    model.n_iterations = model_data.get('n_iterations', 0)
    
    # Възстановява историята на загубите, ако съществува
    if hasattr(model, 'loss_history') and 'loss_history' in model_data:
        model.loss_history = model_data['loss_history']
    
    print(f"Моделът е зареден от {path}")
    return model
