import numpy as np
from PIL import Image, ImageDraw
import random

def create_synthetic_dataset(n_samples_per_class=1000, random_seed=42):
    """
    Create a synthetic dataset of grayscale 28x28 images containing either 
    a white circle or a white square on a black background.
    
    Parameters:
    -----------
    n_samples_per_class : int, default=1000
        Number of samples to generate for each class (circle and square)
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X : numpy.ndarray
        Image data with shape (n_samples, 784) where n_samples = 2 * n_samples_per_class
        Pixel values are normalized between 0 and 1
    y : numpy.ndarray
        Labels with shape (n_samples,) where 0 = circle, 1 = square
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Image dimensions
    img_size = 28
    total_samples = 2 * n_samples_per_class
    
    # Initialize arrays
    X = np.zeros((total_samples, img_size * img_size))  # Flattened images
    y = np.zeros(total_samples, dtype=int)
    
    # Generate circles (label 0)
    for i in range(n_samples_per_class):
        # Create black background
        img = Image.new('L', (img_size, img_size), 0)
        draw = ImageDraw.Draw(img)
        
        # Random circle parameters
        center_x = random.randint(8, 20)
        center_y = random.randint(8, 20)
        radius = random.randint(6, 10)
        
        # Draw white circle
        bbox = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        draw.ellipse(bbox, fill=255)
        
        # Convert to numpy array and flatten
        img_array = np.array(img, dtype=np.float32)
        X[i] = img_array.flatten()
        y[i] = 0  # Circle label
    
    # Generate squares (label 1)
    for i in range(n_samples_per_class):
        # Create black background
        img = Image.new('L', (img_size, img_size), 0)
        draw = ImageDraw.Draw(img)
        
        # Random square parameters
        size = random.randint(8, 16)
        x_pos = random.randint(4, img_size - size - 4)
        y_pos = random.randint(4, img_size - size - 4)
        
        # Draw white square
        bbox = (x_pos, y_pos, x_pos + size, y_pos + size)
        draw.rectangle(bbox, fill=255)
        
        # Convert to numpy array and flatten
        img_array = np.array(img, dtype=np.float32)
        X[n_samples_per_class + i] = img_array.flatten()
        y[n_samples_per_class + i] = 1  # Square label
    
    # Normalize pixel values between 0 and 1
    X = X / 255.0
    
    return X, y

# Example usage and testing
if __name__ == "__main__":
    # Create dataset with 500 samples per class
    X, y = create_synthetic_dataset(n_samples_per_class=500, random_seed=123)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Pixel value range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Number of circles (label 0): {np.sum(y == 0)}")
    print(f"Number of squares (label 1): {np.sum(y == 1)}")
    
    # Visualize a few examples
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    # Show some circles
    circle_indices = np.where(y == 0)[0][:4]
    for i, idx in enumerate(circle_indices):
        img = X[idx].reshape(28, 28)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Circle (label 0)')
        axes[0, i].axis('off')
    
    # Show some squares
    square_indices = np.where(y == 1)[0][:4]
    for i, idx in enumerate(square_indices):
        img = X[idx].reshape(28, 28)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title(f'Square (label 1)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show() 