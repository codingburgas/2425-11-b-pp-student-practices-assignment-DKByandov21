import numpy as np
from PIL import Image, ImageDraw
import random

def test_synthetic():
    print("Testing synthetic dataset creation...")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    
    # Image dimensions
    img_size = 28
    n_samples_per_class = 100
    total_samples = 2 * n_samples_per_class
    
    # Initialize arrays
    X = np.zeros((total_samples, img_size * img_size))
    y = np.zeros(total_samples, dtype=int)
    
    print(f"Created arrays: X shape {X.shape}, y shape {y.shape}")
    
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
        
        if i == 0:
            print(f"First circle: center=({center_x},{center_y}), radius={radius}")
    
    print("Generated circles successfully")
    
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
        
        if i == 0:
            print(f"First square: pos=({x_pos},{y_pos}), size={size}")
    
    print("Generated squares successfully")
    
    # Normalize pixel values between 0 and 1
    X = X / 255.0
    
    print(f"Final dataset: X shape {X.shape}, y shape {y.shape}")
    print(f"Pixel range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Labels: {np.bincount(y)}")
    
    return X, y

if __name__ == "__main__":
    X, y = test_synthetic()
    print("Test completed successfully!") 