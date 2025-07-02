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
    Обучение на Персептрон и Логистична регресия и сравнение на тяхната ефективност.
    """
    print("=== Обучение и сравнение на модели ===\n")
    
    # 1. Създаване на синтетичен набор от данни
    print("1. Създаване на синтетичен набор от данни...")
    X, y = create_synthetic_dataset(n_samples_per_class=1000, random_seed=42)
    print(f"   Размер на набора: X={X.shape}, y={y.shape}")
    print(f"   Брой кръгове (клас 0): {np.sum(y == 0)}")
    print(f"   Брой квадрати (клас 1): {np.sum(y == 1)}")
    
    # 2. Разделяне на данните на тренировъчни и тестови (80/20)
    print("\n2. Разделяне на данните на тренировка/тест (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Тренировъчен набор: {X_train.shape[0]} примера")
    print(f"   Тестов набор: {X_test.shape[0]} примера")
    
    # 3. Обучение на Персептрон
    print("\n3. Обучение на Персептрон...")
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train, epochs=50, learning_rate=0.01)
    
    # 4. Обучение на Логистична регресия
    print("\n4. Обучение на Логистична регресия...")
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train, epochs=100, learning_rate=0.1, verbose=True)
    
    # 5. Оценка на двата модела
    print("\n5. Оценка на моделите...")
    
    perceptron_pred = perceptron.predict(X_test)
    perceptron_metrics = ModelMetrics.calculate_all_metrics(y_test, perceptron_pred)
    
    logistic_pred = logistic_model.predict(X_test)
    logistic_proba = logistic_model.predict_proba(X_test)
    logistic_metrics = ModelMetrics.calculate_all_metrics(y_test, logistic_pred, logistic_proba)
    
    # Сравнение на метриките
    print("\n   === СРАВНЕНИЕ НА МОДЕЛИТЕ ===")
    print(f"   {'Метрика':<15} {'Персептрон':<12} {'Логистична Рег.':<12}")
    print(f"   {'-'*40}")
    print(f"   {'Точност':<15} {perceptron_metrics['accuracy']:<12.4f} {logistic_metrics['accuracy']:<12.4f}")
    print(f"   {'Прецизност':<15} {perceptron_metrics['precision']:<12.4f} {logistic_metrics['precision']:<12.4f}")
    print(f"   {'Чувствителност':<15} {perceptron_metrics['recall']:<12.4f} {logistic_metrics['recall']:<12.4f}")
    print(f"   {'F1-стойност':<15} {perceptron_metrics['f1_score']:<12.4f} {logistic_metrics['f1_score']:<12.4f}")
    if 'log_loss' in logistic_metrics:
        print(f"   {'Log Loss':<15} {'N/A':<12} {logistic_metrics['log_loss']:<12.4f}")
    if 'log_loss' in perceptron_metrics:
        print(f"   {'Log Loss':<15} {'N/A':<12} {perceptron_metrics['log_loss']:<12.4f}")
    
    # 6. Анализ на информационна печалба
    print("\n6. Анализ на важността на пикселите чрез Information Gain...")
    ig_analysis = InformationGain.analyze_feature_importance(X_train, y_train, top_k=10)
    
    print("   Топ 10 най-информативни пиксела:")
    for i, feature in enumerate(ig_analysis['top_features']):
        print(f"   {i+1:2d}. {feature['feature_name']:<15} IG: {feature['information_gain']:.4f}")
    
    # 7. Запазване на моделите
    print("\n7. Запазване на моделите...")
    model_directory = 'app/ai/save_models'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_path = os.path.join(model_directory, "logistic_model.joblib")
    save_model(logistic_model, model_path)
    
    # 8. Визуализация на графики от обучението
    print("\n8. Визуализация на графики от обучението...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Загуба при логистична регресия
    axes[0].plot(range(1, len(logistic_model.loss_history) + 1), 
                logistic_model.loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Епоха')
    axes[0].set_ylabel('Log Loss')
    axes[0].set_title('Загуба при логистична регресия')
    axes[0].grid(True, alpha=0.3)
    
    # Топ 100 пиксела - теплинна карта
    top_100_features = ig_analysis['top_features'][:100]
    importance_map = np.zeros(784)
    for feature in top_100_features:
        importance_map[feature['feature_idx']] = feature['information_gain']
    
    importance_heatmap = importance_map.reshape(28, 28)
    im = axes[1].imshow(importance_heatmap, cmap='hot', interpolation='nearest')
    axes[1].set_title('Теплинна карта на важността на пикселите\n(Топ 100)')
    axes[1].set_xlabel('Колона на пиксела')
    axes[1].set_ylabel('Ред на пиксела')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    # 9. Визуализация на матрици на обърквания
    print("\n9. Визуализация на матрици на обърквания...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Персептрон - объркваща матрица
    cm_perceptron = perceptron_metrics['confusion_matrix']
    im1 = axes[0].imshow(cm_perceptron, interpolation='nearest', cmap='Blues')
    axes[0].set_title('Персептрон\nМатрица на обърквания')
    axes[0].set_xlabel('Предсказан клас')
    axes[0].set_ylabel('Истински клас')
    
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm_perceptron[i, j]), 
                        ha="center", va="center", fontsize=16)
    
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Circle', 'Square'])
    axes[0].set_yticklabels(['Circle', 'Square'])
    
    # Логистична регресия - объркваща матрица
    cm_logistic = logistic_metrics['confusion_matrix']
    im2 = axes[1].imshow(cm_logistic, interpolation='nearest', cmap='Greens')
    axes[1].set_title('Логистична регресия\nМатрица на обърквания')
    axes[1].set_xlabel('Предсказан клас')
    axes[1].set_ylabel('Истински клас')
    
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
    
    print("\n=== Обучението и сравнението са завършени ===")
    return {
        'perceptron': perceptron,
        'logistic': logistic_model,
        'perceptron_metrics': perceptron_metrics,
        'logistic_metrics': logistic_metrics,
        'feature_importance': ig_analysis
    }

if __name__ == "__main__":
    results = train_and_compare_models()
