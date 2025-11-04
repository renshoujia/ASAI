import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, io, color
from skimage.transform import resize

# Load image
image = io.imread('your_image_path_here')  # Replace with your image path
gray_image = color.rgb2gray(image)  # Convert to grayscale

# Resize the image to a smaller size for faster processing (optional)
gray_image = resize(gray_image, (256, 256), anti_aliasing=True)

# Compute GLCM
glcm = feature.greycomatrix((gray_image * 255).astype(np.uint8),
                            distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)

# Compute Texture Features from GLCM
contrast = feature.greycoprops(glcm, prop='contrast')
homogeneity = feature.greycoprops(glcm, prop='homogeneity')
entropy = -np.sum(glcm * np.log2(glcm + 1e-10), axis=(0, 1))

# Display results
print("Contrast:\n", contrast)
print("Homogeneity:\n", homogeneity)
print("Entropy:\n", entropy)

# Optional: Visualize the original and texture features
plt.figure(figsize=(10, 7))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

# Visualize the computed texture features
plt.subplot(1, 2, 2)
plt.title("GLCM Texture Features")
plt.imshow(contrast[0], cmap='hot')  # Example with contrast
plt.axis('off')

plt.show()
