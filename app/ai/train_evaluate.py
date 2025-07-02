import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from synthetic_dataset import create_synthetic_dataset
from perceptron import Perceptron
from model_utils import save_model, load_model
import os

def train_and_evaluate():
    """
    Обучение на модела Персептрон и оценка на неговата ефективност.
    """
    print("=== Обучение и оценка на Персептрон ===\n")
    
    # 1. Създаване на синтетичен набор от данни
    print("1. Създаване на синтетичен набор от данни...")
    X, y = create_synthetic_dataset(n_samples_per_class=1000, random_seed=42)
    print(f"   Размер на набора: X={X.shape}, y={y.shape}")
    print(f"   Брой кръгове (клас 0): {np.sum(y == 0)}")
    print(f"   Брой квадрати (клас 1): {np.sum(y == 1)}")
    
    # 2. Разделяне на данните на тренировъчни и тестови (80/20)
    print("\n2. Разделяне на данните (тренировка/тест) в съотношение 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Тренировъчен набор: {X_train.shape[0]} примера")
    print(f"   Тестов набор: {X_test.shape[0]} примера")
    
    # 3. Създаване и обучение на Персептрон с проследяване на точността
    print("\n3. Обучение на модела Персептрон...")
    perceptron = Perceptron()
    
    # Параметри за обучение
    epochs = 50
    learning_rate = 0.01
    train_accuracies = []
    
    print(f"   Обучение за {epochs} епохи с learning rate {learning_rate}")
    
    # Инициализация на теглата
    n_features = X_train.shape[1]
    weights = np.random.uniform(-0.5, 0.5, n_features + 1)
    
    # Добавяне на bias термин
    X_train_with_bias = np.column_stack([X_train, np.ones(X_train.shape[0])])
    
    for epoch in range(epochs):
        misclassified_count = 0
        
        for i in range(X_train_with_bias.shape[0]):
            z = np.dot(X_train_with_bias[i], weights)
            prediction = 1 if z >= 0 else 0
            
            if prediction != y_train[i]:
                misclassified_count += 1
                update = learning_rate * (y_train[i] - prediction)
                weights += update * X_train_with_bias[i]
        
        # Изчисляване на точността след всяка епоха
        z_train = np.dot(X_train_with_bias, weights)
        train_predictions = np.where(z_train >= 0, 1, 0)
        train_accuracy = np.mean(train_predictions == y_train)
        train_accuracies.append(train_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"   Епоха {epoch + 1:2d}: Точност = {train_accuracy:.4f}, "
                  f"Грешни = {misclassified_count}")
        
        if misclassified_count == 0:
            print(f"   Конвергира след {epoch + 1} епохи!")
            break
    
    # Записване на обучените тегла в обекта
    perceptron.weights = weights
    perceptron.bias = weights[-1]
    perceptron.n_features = n_features
    perceptron.n_iterations = len(train_accuracies)
    
    # 4. Визуализация на точността
    print("\n4. Визуализация на точността по време на обучението...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', linewidth=2)
    plt.xlabel('Епоха')
    plt.ylabel('Тренировъчна точност')
    plt.title('Точност на Персептрон по време на обучение')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Перфектна точност')
    plt.legend()
    
    if len(train_accuracies) < epochs:
        plt.axvline(x=len(train_accuracies), color='g', linestyle=':', 
                   alpha=0.7, label=f'Конвергенция (епоха {len(train_accuracies)})')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 5. Оценка върху тестовия набор
    print("\n5. Оценка върху тестовия набор...")
    test_accuracy = perceptron.score(X_test, y_test)
    train_accuracy_final = perceptron.score(X_train, y_train)
    
    print(f"   Финална тренировъчна точност: {train_accuracy_final:.4f}")
    print(f"   Точност на тестовия набор: {test_accuracy:.4f}")
    print(f"   Общо епохи на обучение: {perceptron.n_iterations}")
    
    # 6. Запазване и зареждане на модела
    print("\n6. Запазване и зареждане на модела...")
    model_directory = 'app/ai/save_models'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    model_path = os.path.join(model_directory, "perceptron_model.joblib")
    save_model(perceptron, model_path)
    
    print("   Зареждане на модела от диск...")
    loaded_perceptron = load_model(model_path, Perceptron)
    
    preds_original = perceptron.predict(X_test)
    preds_loaded = loaded_perceptron.predict(X_test)
    identical = np.array_equal(preds_original, preds_loaded)
    print(f"   Съвпадат ли предсказанията след зареждане: {identical}")
    
    # 7. Подробен анализ
    print("\n7. Подробен анализ:")
    from sklearn.metrics import confusion_matrix, classification_report
    test_predictions = loaded_perceptron.predict(X_test)
    cm = confusion_matrix(y_test, test_predictions)
    
    print(f"   Матрица на объркванията:")
    print(f"   [[{cm[0,0]:4d} {cm[0,1]:4d}]")
    print(f"    [{cm[1,0]:4d} {cm[1,1]:4d}]]")
    
    print(f"\n   Класификационен отчет:")
    print(classification_report(y_test, test_predictions, 
                               target_names=['Circle', 'Square']))
    
    # 8. Визуализация на предсказанията
    print("\n8. Визуализация на предсказанията...")
    
    correct_circles = np.where((test_predictions == y_test) & (y_test == 0))[0][:4]
    correct_squares = np.where((test_predictions == y_test) & (y_test == 1))[0][:4]
    incorrect_examples = np.where(test_predictions != y_test)[0][:4]
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    for i, idx in enumerate(correct_circles):
        img = X_test[idx].reshape(28, 28)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Circle ✓ (Истина: 0, Предсказание: 0)')
        axes[0, i].axis('off')
    
    for i, idx in enumerate(correct_squares):
        img = X_test[idx].reshape(28, 28)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Square ✓ (Истина: 1, Предсказание: 1)')
        axes[1, i].axis('off')
    
    for i, idx in enumerate(incorrect_examples):
        img = X_test[idx].reshape(28, 28)
        true_label = y_test[idx]
        pred_label = test_predictions[idx]
        axes[2, i].imshow(img, cmap='gray')
        axes[2, i].set_title(f'✗ (Истина: {true_label}, Предсказание: {pred_label})')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Обучението и оценката приключиха ===")
    print(f"Точност върху тестовите данни: {perceptron.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_and_evaluate()
