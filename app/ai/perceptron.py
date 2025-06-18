import numpy as np

class Perceptron:
    """
    Binary Perceptron classifier implementation from scratch using NumPy.
    
    The Perceptron is a simple linear classifier that learns to separate two classes
    using a hyperplane. It uses the step activation function and updates weights
    only on misclassified examples.
    
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
    """
    
    def __init__(self):
        """Initialize the Perceptron classifier."""
        self.weights = None
        self.bias = None
        self.n_features = None
        self.n_iterations = 0
    
    def _step_function(self, z):
        """
        Step activation function.
        
        Parameters:
        -----------
        z : numpy.ndarray
            Input to the activation function
            
        Returns:
        --------
        numpy.ndarray
            Output after applying step function (0 or 1)
        """
        return np.where(z >= 0, 1, 0)
    
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
    
    def fit(self, X, y, epochs=10, learning_rate=0.01):
        """
        Fit the Perceptron model to the training data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training features with shape (n_samples, n_features)
        y : numpy.ndarray
            Target labels with shape (n_samples,) containing 0s and 1s
        epochs : int, default=10
            Number of training epochs
        learning_rate : float, default=0.01
            Learning rate for weight updates
            
        Returns:
        --------
        self : Perceptron
            Fitted model instance
        """
        # Input validation
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0s and 1s")
        
        # Initialize weights and bias
        self.n_features = X.shape[1]
        # Initialize weights randomly between -0.5 and 0.5
        self.weights = np.random.uniform(-0.5, 0.5, self.n_features + 1)
        self.bias = self.weights[-1]  # Store bias separately for convenience
        
        # Add bias term to features
        X_with_bias = self._add_bias(X)
        
        # Training loop
        for epoch in range(epochs):
            misclassified_count = 0
            
            # Iterate through all training examples
            for i in range(X_with_bias.shape[0]):
                # Compute prediction
                z = np.dot(X_with_bias[i], self.weights)
                prediction = self._step_function(z)
                
                # Check if example is misclassified
                if prediction != y[i]:
                    misclassified_count += 1
                    
                    # Update weights using Perceptron update rule
                    # w = w + learning_rate * (y_true - y_pred) * x
                    update = learning_rate * (y[i] - prediction)
                    self.weights += update * X_with_bias[i]
                    self.bias = self.weights[-1]  # Update bias
            
            # Early stopping if no misclassifications
            if misclassified_count == 0:
                print(f"Converged after {epoch + 1} epochs")
                break
        
        self.n_iterations = epoch + 1
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features with shape (n_samples, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels (0 or 1) with shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")
        
        # Add bias term to features
        X_with_bias = self._add_bias(X)
        
        # Compute predictions
        z = np.dot(X_with_bias, self.weights)
        predictions = self._step_function(z)
        
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


# Example usage and testing
if __name__ == "__main__":
    # Import the synthetic dataset function
    from synthetic_dataset import create_synthetic_dataset
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    X, y = create_synthetic_dataset(n_samples_per_class=500, random_seed=42)
    
    # Split into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Create and train Perceptron
    print("\nTraining Perceptron...")
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train, epochs=20, learning_rate=0.01)
    
    # Evaluate model
    train_score = perceptron.score(X_train, y_train)
    test_score = perceptron.score(X_test, y_test)
    
    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    print(f"Number of training iterations: {perceptron.n_iterations}")
    
    # Get learned weights
    feature_weights, bias = perceptron.get_weights()
    print(f"Bias term: {bias:.4f}")
    print(f"Weight vector shape: {feature_weights.shape}")
    
    # Visualize some predictions
    import matplotlib.pyplot as plt
    
    # Get some test predictions
    test_predictions = perceptron.predict(X_test)
    
    # Find some correctly and incorrectly classified examples
    correct_mask = test_predictions == y_test
    incorrect_mask = ~correct_mask
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Show some correct predictions
    correct_indices = np.where(correct_mask)[0][:4]
    for i, idx in enumerate(correct_indices):
        img = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = test_predictions[idx]
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'True: {true_label}, Pred: {pred_label}')
        axes[0, i].axis('off')
    
    # Show some incorrect predictions
    incorrect_indices = np.where(incorrect_mask)[0][:4]
    for i, idx in enumerate(incorrect_indices):
        img = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = test_predictions[idx]
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'True: {true_label}, Pred: {pred_label}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show() 