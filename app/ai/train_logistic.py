
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from synthetic_dataset import create_synthetic_dataset
from logistic_regression import LogisticRegression
from perceptron import Perceptron
from model_utils import save_model, load_model
from metrics import ModelMetrics, InformationGain
import os

def train_and_compare_models():
    """
    Train both Perceptron and Logistic Regression models and compare their performance.
    """
    print("=== Model Training and Comparison ===\n")
    
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
    
    # 3. Train Perceptron model
    print("\n3. Training Perceptron model...")
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train, epochs=50, learning_rate=0.01)
    
    # 4. Train Logistic Regression model
    print("\n4. Training Logistic Regression model...")
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train, epochs=100, learning_rate=0.1, verbose=True)
    
    # 5. Evaluate both models
    print("\n5. Evaluating models...")
    
    # Perceptron predictions
    perceptron_pred = perceptron.predict(X_test)
    perceptron_metrics = ModelMetrics.calculate_all_metrics(y_test, perceptron_pred)
    
    # Logistic Regression predictions
    logistic_pred = logistic_model.predict(X_test)
    logistic_proba = logistic_model.predict_proba(X_test)
    logistic_metrics = ModelMetrics.calculate_all_metrics(y_test, logistic_pred, logistic_proba)
    
    # Print comparison
    print("\n   === MODEL COMPARISON ===")
    print(f"   {'Metric':<15} {'Perceptron':<12} {'Logistic Reg':<12}")
    print(f"   {'-'*40}")
    print(f"   {'Accuracy':<15} {perceptron_metrics['accuracy']:<12.4f} {logistic_metrics['accuracy']:<12.4f}")
    print(f"   {'Precision':<15} {perceptron_metrics['precision']:<12.4f} {logistic_metrics['precision']:<12.4f}")
    print(f"   {'Recall':<15} {perceptron_metrics['recall']:<12.4f} {logistic_metrics['recall']:<12.4f}")
    print(f"   {'F1-Score':<15} {perceptron_metrics['f1_score']:<12.4f} {logistic_metrics['f1_score']:<12.4f}")
    if 'log_loss' in logistic_metrics:
        print(f"   {'Log Loss':<15} {'N/A':<12} {logistic_metrics['log_loss']:<12.4f}")
    
    # 6. Information Gain Analysis
    print("\n6. Analyzing feature importance with Information Gain...")
    ig_analysis = InformationGain.analyze_feature_importance(X_train, y_train, top_k=10)
    
    print("   Top 10 most informative pixels:")
    for i, feature in enumerate(ig_analysis['top_features']):
        print(f"   {i+1:2d}. {feature['feature_name']:<15} IG: {feature['information_gain']:.4f}")
    
    # 7. Save models
    print("\n7. Saving models...")
    save_model(perceptron, "perceptron_model.joblib")
    save_model(logistic_model, "logistic_model.joblib")
    
    # 8. Plot training curves
    print("\n8. Plotting training curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot logistic regression loss curve
    axes[0].plot(range(1, len(logistic_model.loss_history) + 1), 
                logistic_model.loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Log Loss')
    axes[0].set_title('Logistic Regression Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Plot feature importance heatmap (top 100 pixels)
    top_100_features = ig_analysis['top_features'][:100]
    importance_map = np.zeros(784)
    for feature in top_100_features:
        importance_map[feature['feature_idx']] = feature['information_gain']
    
    importance_heatmap = importance_map.reshape(28, 28)
    im = axes[1].imshow(importance_heatmap, cmap='hot', interpolation='nearest')
    axes[1].set_title('Feature Importance Heatmap\n(Top 100 Pixels by Information Gain)')
    axes[1].set_xlabel('Pixel Column')
    axes[1].set_ylabel('Pixel Row')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    # 9. Visualize confusion matrices
    print("\n9. Plotting confusion matrices...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Perceptron confusion matrix
    cm_perceptron = perceptron_metrics['confusion_matrix']
    im1 = axes[0].imshow(cm_perceptron, interpolation='nearest', cmap='Blues')
    axes[0].set_title('Perceptron\nConfusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm_perceptron[i, j]), 
                        ha="center", va="center", fontsize=16)
    
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Circle', 'Square'])
    axes[0].set_yticklabels(['Circle', 'Square'])
    
    # Logistic Regression confusion matrix
    cm_logistic = logistic_metrics['confusion_matrix']
    im2 = axes[1].imshow(cm_logistic, interpolation='nearest', cmap='Greens')
    axes[1].set_title('Logistic Regression\nConfusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm_logistic[i, j]), 
                        ha="center", va="center", fontsize=16)
    
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Circle', 'Square'])
    axes[1].set_yticklabels(['Circle', 'Square'])
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Training and Comparison Complete ===")
    return {
        'perceptron': perceptron,
        'logistic': logistic_model,
        'perceptron_metrics': perceptron_metrics,
        'logistic_metrics': logistic_metrics,
        'feature_importance': ig_analysis
    }

if __name__ == "__main__":
    results = train_and_compare_models()
