import numpy as np

class LogisticRegression:
    
    def __init__(self, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        """Инициализиране на логистичния регресионен класификатор."""
        self.weights = None
        self.bias = None
        self.n_features = None
        self.n_iterations = 0
        self.loss_history = []
    
    def _sigmoid(self, z):
        """
        Сигмоидна активационна функция с числова стабилност.
        Сигмоидна стойност между 0 и 1
        """
        # Ограничаване на z за предотвратяване на препълване
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _add_bias(self, X):
        """
        Добавя bias термин към входните характеристики.
        """
        return np.column_stack([X, np.ones(X.shape[0])])
    
    def _compute_loss(self, y_true, y_pred):
        """
        Изчислява крос-ентропийна (логаритмична) загуба.

        y_true : numpy.ndarray
            Истински бинарни етикети
        y_pred : numpy.ndarray
            Предсказани вероятности
        """
        # Малко епсилон за предотвратяване на log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y, epochs=100, learning_rate=0.01, verbose=False):

        if X.shape[0] != y.shape[0]:
            raise ValueError("X и y трябва да имат еднакъв брой примери")
        
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y трябва да съдържа само 0 и 1")
        
        self.n_features = X.shape[1]
        # Инициализация на теглата с Xavier метод
        self.weights = np.random.normal(0, np.sqrt(2.0 / self.n_features), self.n_features + 1)
        self.bias = self.weights[-1]
        
        X_with_bias = self._add_bias(X)
        self.loss_history = []
        
        # Тренировъчен цикъл с градиентен спад
        for epoch in range(epochs):
            z = np.dot(X_with_bias, self.weights)
            predictions = self._sigmoid(z)
            
            loss = self._compute_loss(y, predictions)
            self.loss_history.append(loss)
            
            error = predictions - y
            gradients = np.dot(X_with_bias.T, error) / X.shape[0]
            
            self.weights -= learning_rate * gradients
            self.bias = self.weights[-1]
            
            if verbose and (epoch + 1) % 20 == 0:
                accuracy = self.score(X, y)
                print(f"Епоха {epoch + 1:3d}: Загуба = {loss:.6f}, Точност = {accuracy:.4f}")
        
        self.n_iterations = epochs
        return self
    
    def predict_proba(self, X):
        """
        Предсказва вероятности за клас 1 за подадените примери.
        """
        if self.weights is None:
            raise ValueError("Моделът трябва да бъде обучен преди предсказване")
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"Очаквани {self.n_features} характеристики, получени {X.shape[1]}")
        
        X_with_bias = self._add_bias(X)
        z = np.dot(X_with_bias, self.weights)
        probabilities = self._sigmoid(z)
        
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """
        Предсказва класови етикети (0 или 1) на база вероятности.
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        return predictions
    
    def score(self, X, y):
        """
        Изчислява точността на модела върху подадените данни.
        Точност (accuracy) между 0 и 1
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def get_weights(self):
        """
        Връща научените тегла и bias.
        
        """
        if self.weights is None:
            raise ValueError("Моделът трябва да бъде обучен преди извличане на теглата")
        
        feature_weights = self.weights[:-1]  # Всички без bias
        return feature_weights, self.bias
    
    def get_feature_importance(self, feature_names=None):
        """
        Връща важността на характеристиките въз основа на абсолютните стойности на теглата.
        """
        if self.weights is None:
            raise ValueError("Моделът трябва да бъде обучен преди извличане на важностите")
        
        feature_weights = np.abs(self.weights[:-1])  # Без bias
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(feature_weights))]
        
        importance_dict = dict(zip(feature_names, feature_weights))
        
        sorted_importance = dict(sorted(importance_dict.items(), 
                                        key=lambda x: x[1], reverse=True))
        
        return sorted_importance
