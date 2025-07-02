import numpy as np

class Perceptron:
    """
    Имплементация на бинарен персептрон класификатор от нулата с помощта на NumPy.
    """
    
    def __init__(self, learning_rate=0.01, max_epochs=10):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.n_features = None
        self.n_iterations = 0
    
    def _step_function(self, z):
        """
        Стъпкова (step) активационна функция.
        """
        return np.where(z >= 0, 1, 0)
    
    def _add_bias(self, X):
        """
        Добавя bias термин към входните характеристики.
        """
        return np.column_stack([X, np.ones(X.shape[0])])
    
    def fit(self, X, y):
        epochs = self.max_epochs
        learning_rate = self.learning_rate
        """
        Обучава модела Персептрон с тренировъчните данни.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X и y трябва да имат еднакъв брой примери")
        
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y трябва да съдържа само 0 и 1")
        
        self.n_features = X.shape[1]
        self.weights = np.random.uniform(-0.5, 0.5, self.n_features + 1)
        self.bias = self.weights[-1]
        X_with_bias = self._add_bias(X)
        
        for epoch in range(epochs):
            misclassified_count = 0
            
            for i in range(X_with_bias.shape[0]):
                z = np.dot(X_with_bias[i], self.weights)
                prediction = self._step_function(z)
                
                if prediction != y[i]:
                    misclassified_count += 1
                    update = learning_rate * (y[i] - prediction)
                    self.weights += update * X_with_bias[i]
                    self.bias = self.weights[-1]
            
            if misclassified_count == 0:
                print(f"Конвергира след {epoch + 1} епохи")
                break
        
        self.n_iterations = epoch + 1
        return self
    
    def predict(self, X):
        """
        Предсказва класовите етикети (0 или 1) за нови примери.
        
        """
        if self.weights is None:
            raise ValueError("Моделът трябва да бъде обучен преди предсказване")
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"Очаквани {self.n_features} характеристики, получени {X.shape[1]}")
        
        X_with_bias = self._add_bias(X)
        z = np.dot(X_with_bias, self.weights)
        predictions = self._step_function(z)
        return predictions
    
    def score(self, X, y):
        """
        Изчислява точността на модела спрямо дадени данни.

        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def get_weights(self):
        """
        Връща научените тегла и bias.
        
        """
        if self.weights is None:
            raise ValueError("Моделът трябва да бъде обучен преди достъп до теглата")
        
        feature_weights = self.weights[:-1]
        return feature_weights, self.bias


# Пример за използване и тестване на модела
if __name__ == "__main__":
    from synthetic_dataset import create_synthetic_dataset
    
    print("Създаване на синтетичен набор от данни...")
    X, y = create_synthetic_dataset(n_samples_per_class=500, random_seed=42)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Форма на тренировъчния набор: {X_train.shape}")
    print(f"Форма на тестовия набор: {X_test.shape}")
    
    print("\nОбучение на персептрон...")
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train, epochs=20, learning_rate=0.01)
    
    train_score = perceptron.score(X_train, y_train)
    test_score = perceptron.score(X_test, y_test)
    
    print(f"\nТочност на тренировъчния набор: {train_score:.4f}")
    print(f"Точност на тестовия набор: {test_score:.4f}")
    print(f"Брой тренировъчни итерации: {perceptron.n_iterations}")
    
    feature_weights, bias = perceptron.get_weights()
    print(f"Bias термин: {bias:.4f}")
    print(f"Форма на вектора с тегла: {feature_weights.shape}")
    
    # Визуализация на предсказания
    import matplotlib.pyplot as plt
    
    test_predictions = perceptron.predict(X_test)
    correct_mask = test_predictions == y_test
    incorrect_mask = ~correct_mask
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Визуализация на правилни предсказания
    correct_indices = np.where(correct_mask)[0][:4]
    for i, idx in enumerate(correct_indices):
        img = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = test_predictions[idx]
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Истински: {true_label}, Предсказан: {pred_label}')
        axes[0, i].axis('off')
    
    # Визуализация на грешни предсказания
    incorrect_indices = np.where(incorrect_mask)[0][:4]
    for i, idx in enumerate(incorrect_indices):
        img = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = test_predictions[idx]
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Истински: {true_label}, Предсказан: {pred_label}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
