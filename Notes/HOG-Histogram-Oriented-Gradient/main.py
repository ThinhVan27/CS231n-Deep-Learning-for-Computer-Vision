import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_image(image, title="Image"):
    """Display the image with a title."""
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def rgb2gray(image):
    """Convert an RGB image to grayscale."""
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def calc_gradient(image):
    """Calculate the gradient of the image."""
    
    Dx = np.array([[-1, 0, 1]], dtype=np.float32)
    Dy = np.array([[-1], [0], [1]], dtype=np.float32)
    
    # Calculate gradients in x and y directions
    gradient_x = cv2.filter2D(image.astype(np.float32), -1, Dx)
    gradient_y = cv2.filter2D(image.astype(np.float32), -1, Dy)
    
    # Gradient Magnitude and Direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan(gradient_y, gradient_x) * 180 / np.pi + 90 # Normalize to [0, 180)
    
    return gradient_x, gradient_y, gradient_magnitude, gradient_direction

def show_gradient(gradient_x, gradient_y, gradient_magnitude):
    """Display the gradient magnitude and direction."""
    plt.subplot(1, 3, 1)
    plt.imshow(gradient_x, cmap='gray')
    plt.title('Gradient X')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(gradient_y, cmap='gray')
    plt.title('Gradient Y')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')
    
    plt.show()
def calc_histogram(gradient_direction, gradient_magnitude):
    """Calculate the histogram of gradient directions."""
    hist = np.zeros(shape=(16, 8, 9), dtype=np.float32) # 8 pixels per cell, 9 bins
    
    for i in range(0, gradient_direction.shape[0], 8):
        for j in range(0, gradient_direction.shape[1], 8):
            cell_dir = gradient_direction[i:i+8, j:j+8]
            cell_mag = gradient_magnitude[i:i+8, j:j+8]
            hist_cell = np.zeros((9, ), dtype=np.float32)
            for k in range(8):
                for l in range(8):
                    first_bin = int(cell_dir[k, l] / 20)
                    if first_bin * 20 == cell_dir[k, l]:
                        hist_cell[first_bin] += cell_mag[k, l]
                    else:
                        hist_cell[first_bin] += (20 * first_bin + 20 - cell_dir[k, l]) * cell_mag[k, l] / 20
                        hist_cell[(first_bin + 1)%9] += cell_mag[k, l] - hist_cell[first_bin]
            hist[i//8, j//8] = hist_cell
    
    return hist

def hog_feature(hist: np.ndarray):
    """Normalize the histogram and concatenate the features"""
    
    features = []
    for i in range(0, hist.shape[0] - 1, 1):
        for j in range(0, hist.shape[1] - 1, 1):
            features.extend(list(hist[i:i+2, j:j+2].reshape(-1) / np.sqrt(np.sum(hist[i:i+2, j:j+2]**2) + 1e-5)))

    return np.array(features)

img = rgb2gray(cv2.resize(cv2.imread("sample.png"), dsize=(64, 128)))

grad_x, grad_y, grad_magnitude, grad_direction = calc_gradient(img)

# Display the gradient magnitude and direction
#show_gradient(grad_x, grad_y, grad_magnitude)

# Calculate the histogram of gradient directions
hist = calc_histogram(grad_direction, grad_magnitude)
print(hist.shape) # 16x8x9

# Calculate the HOG features
hog_features = hog_feature(hist)
print(hog_features.shape) # 3780

