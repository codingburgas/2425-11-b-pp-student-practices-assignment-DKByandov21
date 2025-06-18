import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from synthetic_dataset import create_synthetic_dataset
from perceptron import Perceptron
from model_utils import save_model, load_model
import os

def train_and_evaluate():
    """
    Train the Perceptron model and evaluate its performance.
    """
    print("=== Perceptron Training and Evaluation ===\n")
    
    # 1. Create synthetic dataset
    print("1. Creating synthetic dataset...")
    X, y = create_synthetic_dataset(n_samples_per_class=1000, random_seed=42)
    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Number of circles (class 0): {np.sum(y == 0)}")
    print(f"   Number of squares (class 1): {np.sum(y == 1)}")
    
    # 2. Split dataset into train/test (80/20)
    print("\n2. Splitting dataset into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # 3. Create and train Perceptron with accuracy tracking
    print("\n3. Training Perceptron model...")
    perceptron = Perceptron()
    
    # Training parameters
    epochs = 50
    learning_rate = 0.01
    
    # Track training accuracy over epochs
    train_accuracies = []
    
    # Custom training loop to track accuracy
    print(f"   Training for {epochs} epochs with learning rate {learning_rate}")
    
    # Initialize weights
    n_features = X_train.shape[1]
    weights = np.random.uniform(-0.5, 0.5, n_features + 1)
    
    # Add bias term to training data
    X_train_with_bias = np.column_stack([X_train, np.ones(X_train.shape[0])])
    
    for epoch in range(epochs):
        misclassified_count = 0
        
        # Iterate through all training examples
        for i in range(X_train_with_bias.shape[0]):
            # Compute prediction
            z = np.dot(X_train_with_bias[i], weights)
            prediction = 1 if z >= 0 else 0
            
            # Check if example is misclassified
            if prediction != y_train[i]:
                misclassified_count += 1
                
                # Update weights using Perceptron update rule
                update = learning_rate * (y_train[i] - prediction)
                weights += update * X_train_with_bias[i]
        
        # Calculate training accuracy for this epoch
        # Use current weights to predict on training set
        z_train = np.dot(X_train_with_bias, weights)
        train_predictions = np.where(z_train >= 0, 1, 0)
        train_accuracy = np.mean(train_predictions == y_train)
        train_accuracies.append(train_accuracy)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1:2d}: Training Accuracy = {train_accuracy:.4f}, "
                  f"Misclassified = {misclassified_count}")
        
        # Early stopping if no misclassifications
        if misclassified_count == 0:
            print(f"   Converged after {epoch + 1} epochs!")
            break
    
    # Update the perceptron object with final weights
    perceptron.weights = weights
    perceptron.bias = weights[-1]
    perceptron.n_features = n_features
    perceptron.n_iterations = len(train_accuracies)
    
    # 4. Plot training accuracy over time
    print("\n4. Plotting training accuracy over time...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title('Perceptron Training Accuracy Over Time')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add horizontal line at 1.0 for reference
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Accuracy')
    plt.legend()
    
    # Mark convergence point if early stopping occurred
    if len(train_accuracies) < epochs:
        plt.axvline(x=len(train_accuracies), color='g', linestyle=':', 
                   alpha=0.7, label=f'Convergence (Epoch {len(train_accuracies)})')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 5. Evaluate on test set
    print("\n5. Evaluating on test set...")
    test_accuracy = perceptron.score(X_test, y_test)
    train_accuracy_final = perceptron.score(X_train, y_train)
    
    print(f"   Final Training Accuracy: {train_accuracy_final:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f}")
    print(f"   Total Training Epochs: {perceptron.n_iterations}")
    
    # 6. Save and reload the model
    print("\n6. Saving and reloading the model...")
    model_path = "perceptron_model.joblib"
    save_model(perceptron, model_path)
    
    # Simulate script restart by reloading model and making predictions
    print("   Reloading model from disk...")
    loaded_perceptron = load_model(model_path, Perceptron)
    
    # Check that predictions are identical
    preds_original = perceptron.predict(X_test)
    preds_loaded = loaded_perceptron.predict(X_test)
    identical = np.array_equal(preds_original, preds_loaded)
    print(f"   Predictions identical after reload: {identical}")
    
    # 7. Detailed analysis
    print("\n7. Detailed Analysis:")
    
    # Get predictions on test set
    test_predictions = loaded_perceptron.predict(X_test)
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(y_test, test_predictions)
    print(f"   Confusion Matrix:")
    print(f"   [[{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f"    [{cm[1,0]:4d} {cm[1,1]:4d}]]")
    
    # Print classification report
    print(f"\n   Classification Report:")
    print(classification_report(y_test, test_predictions, 
                               target_names=['Circle', 'Square']))
    
    # 8. Visualize some predictions
    print("\n8. Visualizing predictions...")
    
    # Find some examples of each type
    correct_circles = np.where((test_predictions == y_test) & (y_test == 0))[0][:4]
    correct_squares = np.where((test_predictions == y_test) & (y_test == 1))[0][:4]
    incorrect_examples = np.where(test_predictions != y_test)[0][:4]
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    # Correctly classified circles
    for i, idx in enumerate(correct_circles):
        img = X_test[idx].reshape(28, 28)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Circle ✓ (True: 0, Pred: 0)')
        axes[0, i].axis('off')
    
    # Correctly classified squares
    for i, idx in enumerate(correct_squares):
        img = X_test[idx].reshape(28, 28)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Square ✓ (True: 1, Pred: 1)')
        axes[1, i].axis('off')
    
    # Incorrectly classified examples
    for i, idx in enumerate(incorrect_examples):
        img = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = test_predictions[idx]
        axes[2, i].imshow(img, cmap='gray')
        axes[2, i].set_title(f'✗ (True: {true_label}, Pred: {pred_label})')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Training and Evaluation Complete ===")
    print(f"Final Test Accuracy: {perceptron.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_and_evaluate() 