import numpy as np
from PIL import Image, ImageDraw
import random

def test_synthetic():
    print("Тестване на генерирането на синтетичен набор от данни...")

    # Задаване на начално случайно число (seed) за възпроизводимост
    random.seed(42)
    np.random.seed(42)

    # Размер на изображението
    img_size = 28
    n_samples_per_class = 100
    total_samples = 2 * n_samples_per_class

    # Инициализиране на масиви
    X = np.zeros((total_samples, img_size * img_size))  # Масив за изображенията (разгънати)
    y = np.zeros(total_samples, dtype=int)              # Масив за етикетите

    print(f"Създадени масиви: X форма {X.shape}, y форма {y.shape}")

    # Генериране на кръгове (етикет 0)
    for i in range(n_samples_per_class):
        # Създаване на черен фон
        img = Image.new('L', (img_size, img_size), 0)
        draw = ImageDraw.Draw(img)

        # Случайни параметри за кръг
        center_x = random.randint(8, 20)
        center_y = random.randint(8, 20)
        radius = random.randint(6, 10)

        # Рисуване на бял кръг
        bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        draw.ellipse(bbox, fill=255)

        # Конвертиране в NumPy масив и разгъване
        img_array = np.array(img, dtype=np.float32)
        X[i] = img_array.flatten()
        y[i] = 0  # Етикет за кръг

        if i == 0:
            print(f"Първи кръг: център=({center_x},{center_y}), радиус={radius}")

    print("Кръговете са генерирани успешно")

    # Генериране на квадрати (етикет 1)
    for i in range(n_samples_per_class):
        # Създаване на черен фон
        img = Image.new('L', (img_size, img_size), 0)
        draw = ImageDraw.Draw(img)

        # Случайни параметри за квадрат
        size = random.randint(8, 16)
        x_pos = random.randint(4, img_size - size - 4)
        y_pos = random.randint(4, img_size - size - 4)

        # Рисуване на бял квадрат
        bbox = (x_pos, y_pos, x_pos + size, y_pos + size)
        draw.rectangle(bbox, fill=255)

        # Конвертиране в NumPy масив и разгъване
        img_array = np.array(img, dtype=np.float32)
        X[n_samples_per_class + i] = img_array.flatten()
        y[n_samples_per_class + i] = 1  # Етикет за квадрат

        if i == 0:
            print(f"Първи квадрат: позиция=({x_pos},{y_pos}), размер={size}")

    print("Квадратите са генерирани успешно")

    # Нормализиране на пикселите между 0 и 1
    X = X / 255.0

    print(f"Краен набор от данни: X форма {X.shape}, y форма {y.shape}")
    print(f"Диапазон на стойности на пикселите: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Етикети (брой): {np.bincount(y)}")

    return X, y

if __name__ == "__main__":
    X, y = test_synthetic()
    print("Тестът приключи успешно!")
