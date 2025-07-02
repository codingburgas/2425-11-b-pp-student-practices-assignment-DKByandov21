import numpy as np
from PIL import Image, ImageDraw
import random

def create_synthetic_dataset(n_samples_per_class=1000, random_seed=42):
    """
    Създава синтетичен набор от сиви 28x28 изображения, съдържащи
    бял кръг или бял квадрат на черен фон.
    
    Параметри:
    -----------
    n_samples_per_class : int, по подразбиране=1000
        Брой примери, които да се генерират за всеки клас (кръг и квадрат)
    random_seed : int, по подразбиране=42
        Случаен seed за възпроизводимост
        
    Връща:
    --------
    X : numpy.ndarray
        Данни за изображенията с форма (брой_примери, 784), където брой_примери = 2 * n_samples_per_class.
        Стойностите на пикселите са нормализирани между 0 и 1.
    y : numpy.ndarray
        Етикети с форма (брой_примери,), където 0 = кръг, 1 = квадрат
    """
    
    # Сетване на seed за възпроизводимост
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Размер на изображението
    img_size = 28
    total_samples = 2 * n_samples_per_class
    
    # Инициализация на масивите
    X = np.zeros((total_samples, img_size * img_size))  # Плоски изображения
    y = np.zeros(total_samples, dtype=int)
    
    # Генериране на кръгове (етикет 0)
    for i in range(n_samples_per_class):
        img = Image.new('L', (img_size, img_size), 0)  # Черен фон
        draw = ImageDraw.Draw(img)
        
        # Случайни параметри на кръга
        center_x = random.randint(8, 20)
        center_y = random.randint(8, 20)
        radius = random.randint(6, 10)
        
        # Рисуване на бял кръг
        bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        draw.ellipse(bbox, fill=255)
        
        # Преобразуване към NumPy и flatten
        img_array = np.array(img, dtype=np.float32)
        X[i] = img_array.flatten()
        y[i] = 0  # Етикет за кръг
    
    # Генериране на квадрати (етикет 1)
    for i in range(n_samples_per_class):
        img = Image.new('L', (img_size, img_size), 0)
        draw = ImageDraw.Draw(img)
        
        # Случайни параметри на квадрата
        size = random.randint(8, 16)
        x_pos = random.randint(4, img_size - size - 4)
        y_pos = random.randint(4, img_size - size - 4)
        
        # Рисуване на бял квадрат
        bbox = (x_pos, y_pos, x_pos + size, y_pos + size)
        draw.rectangle(bbox, fill=255)
        
        # Преобразуване към NumPy и flatten
        img_array = np.array(img, dtype=np.float32)
        X[n_samples_per_class + i] = img_array.flatten()
        y[n_samples_per_class + i] = 1  # Етикет за квадрат
    
    # Нормализиране на стойностите на пикселите между 0 и 1
    X = X / 255.0
    
    return X, y

# Примерна употреба и тестване
if __name__ == "__main__":
    # Създаване на набор от данни с 500 примера на клас
    X, y = create_synthetic_dataset(n_samples_per_class=500, random_seed=123)
    
    print(f"Форма на набора от данни: X={X.shape}, y={y.shape}")
    print(f"Диапазон на пикселите: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Брой кръгове (етикет 0): {np.sum(y == 0)}")
    print(f"Брой квадрати (етикет 1): {np.sum(y == 1)}")
    
    # Визуализация на няколко примера
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Показване на кръгове
    circle_indices = np.where(y == 0)[0][:4]
    for i, idx in enumerate(circle_indices):
        img = X[idx].reshape(28, 28)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Кръг (етикет 0)')
        axes[0, i].axis('off')
    
    # Показване на квадрати
    square_indices = np.where(y == 1)[0][:4]
    for i, idx in enumerate(square_indices):
        img = X[idx].reshape(28, 28)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Квадрат (етикет 1)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
