import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, io, color
from skimage.transform import resize

# 加载图像
image = io.imread('your_image_path_here')  # 替换为你的图像路径
gray_image = color.rgb2gray(image)  # 转换为灰度图像

# 可选：调整图像大小以提高处理速度
gray_image = resize(gray_image, (256, 256), anti_aliasing=True)

# 计算GLCM（灰度共生矩阵）
glcm = feature.greycomatrix((gray_image * 255).astype(np.uint8),
                            distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)

# 计算纹理特征
contrast = feature.greycoprops(glcm, prop='contrast')
homogeneity = feature.greycoprops(glcm, prop='homogeneity')
entropy = -np.sum(glcm * np.log2(glcm + 1e-10), axis=(0, 1))

# 计算ASAI指数
asai = (entropy - homogeneity) / (entropy + homogeneity)

# 打印计算结果
print("Contrast:\n", contrast)
print("Homogeneity:\n", homogeneity)
print("Entropy:\n", entropy)
print("ASAI:\n", asai)

# 可选：可视化原始图像和纹理特征
plt.figure(figsize=(10, 7))

# 原始图像
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

# 显示计算的ASAI指数（使用熵的热图作为示例）
plt.subplot(1, 2, 2)
plt.title("ASAI Index (based on Homogeneity and Entropy)")
plt.imshow(asai[0], cmap='hot')  # 示例：显示ASAI热图
plt.axis('off')

plt.show()
