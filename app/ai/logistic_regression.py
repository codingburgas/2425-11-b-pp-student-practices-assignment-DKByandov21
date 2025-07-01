
import numpy as np

class LogisticRegression:
    """
    Binary Logistic Regression classifier implementation from scratch using NumPy.
    
    Uses sigmoid activation function and cross-entropy loss for binary classification.
    Supports gradient descent optimization with configurable learning rate and epochs.
    
    Attributes:
    -----------
    weights : numpy.ndarray
        Weight vector including bias term (last element)
    bias : float
        Bias term (stored separately for convenience)
    n_features : int
        Number of input features
    n_iterations : int
        Number of iterations during training
    loss_history : list
        Training loss history for each epoch
    """
    
    def __init__(self):
        """Initialize the Logistic Regression classifier."""
        self.weights = None
        self.bias = None
        self.n_features = None
        self.n_iterations = 0
        self.loss_history = []
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function with numerical stability.
        
        Parameters:
        -----------
        z : numpy.ndarray
            Input to the sigmoid function
            
        Returns:
        --------
        numpy.ndarray
            Sigmoid output between 0 and 1
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _add_bias(self, X):
        """
        Add bias term to input features.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features with shape (n_samples, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Features with bias term added as last column
        """
        return np.column_stack([X, np.ones(X.shape[0])])
    
    def _compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy (log) loss.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True binary labels
        y_pred : numpy.ndarray
            Predicted probabilities
            
        Returns:
        --------
        float
            Cross-entropy loss
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y, epochs=100, learning_rate=0.01, verbose=False):
        """
        Fit the Logistic Regression model to the training data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training features with shape (n_samples, n_features)
        y : numpy.ndarray
            Target labels with shape (n_samples,) containing 0s and 1s
        epochs : int, default=100
            Number of training epochs
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        verbose : bool, default=False
            Whether to print training progress
            
        Returns:
        --------
        self : LogisticRegression
            Fitted model instance
        """
        # Input validation
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0s and 1s")
        
        # Initialize weights and bias
        self.n_features = X.shape[1]
        # Initialize weights using Xavier initialization
        self.weights = np.random.normal(0, np.sqrt(2.0 / self.n_features), self.n_features + 1)
        self.bias = self.weights[-1]
        
        # Add bias term to features
        X_with_bias = self._add_bias(X)
        
        # Reset loss history
        self.loss_history = []
        
        # Training loop using gradient descent
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X_with_bias, self.weights)
            predictions = self._sigmoid(z)
            
            # Compute loss
            loss = self._compute_loss(y, predictions)
            self.loss_history.append(loss)
            
            # Compute gradients
            error = predictions - y
            gradients = np.dot(X_with_bias.T, error) / X.shape[0]
            
            # Update weights
            self.weights -= learning_rate * gradients
            self.bias = self.weights[-1]
            
            # Print progress
            if verbose and (epoch + 1) % 20 == 0:
                accuracy = self.score(X, y)
                print(f"Epoch {epoch + 1:3d}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}")
        
        self.n_iterations = epochs
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features with shape (n_samples, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Predicted probabilities for class 1 with shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        # Add bias term to features
        X_with_bias = self._add_bias(X)
        
        # Compute probabilities for class 1
        z = np.dot(X_with_bias, self.weights)
        probabilities = self._sigmoid(z)
        
        return probabilities
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features with shape (n_samples, n_features)
        threshold : float, default=0.5
            Decision threshold for binary classification
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels (0 or 1) with shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        return predictions
    
    def score(self, X, y):
        """
        Calculate the accuracy score of the model on the given data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features with shape (n_samples, n_features)
        y : numpy.ndarray
            True labels with shape (n_samples,)
            
        Returns:
        --------
        float
            Accuracy score between 0 and 1
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def get_weights(self):
        """
        Get the learned weights and bias.
        
        Returns:
        --------
        tuple
            (weights, bias) where weights is array of feature weights and bias is scalar
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before accessing weights")
        
        feature_weights = self.weights[:-1]  # All except bias
        return feature_weights, self.bias
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance based on absolute weight values.
        
        Parameters:
        -----------
        feature_names : list, optional
            Names for features. If None, uses indices.
            
        Returns:
        --------
        dict
            Dictionary mapping feature names to importance scores
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before accessing weights")
        
        feature_weights = np.abs(self.weights[:-1])  # Exclude bias
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(feature_weights))]
        
        importance_dict = dict(zip(feature_names, feature_weights))
        
        # Sort by importance (descending)
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
