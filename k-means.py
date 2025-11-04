import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.cluster import KMeans
from skimage.transform import resize

# 加载图像
image = io.imread('your_image_path_here')  # 替换为你的图像路径
gray_image = color.rgb2gray(image)  # 转换为灰度图像

# 可选：调整图像大小以提高处理速度
gray_image = resize(gray_image, (256, 256), anti_aliasing=True)

# 提取图像的纹理特征（例如，使用GLCM）
from skimage import feature
glcm = feature.greycomatrix((gray_image * 255).astype(np.uint8),
                            distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)

# 提取特定的纹理特征：对比度、同质性、熵等
contrast = feature.greycoprops(glcm, prop='contrast')
homogeneity = feature.greycoprops(glcm, prop='homogeneity')
entropy = -np.sum(glcm * np.log2(glcm + 1e-10), axis=(0, 1))

# 重新排列纹理特征（每个像素的特征向量）
features = np.stack([contrast.flatten(), homogeneity.flatten(), entropy.flatten()], axis=1)

# 使用K-means聚类对图像进行分类
kmeans = KMeans(n_clusters=3, random_state=42)  # 假设分为3个聚类
kmeans.fit(features)

# 获取聚类结果
clustered_image = kmeans.labels_.reshape(gray_image.shape)

# 可视化结果
plt.figure(figsize=(12, 6))

# 原始图像
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

# 聚类结果
plt.subplot(1, 2, 2)
plt.title("K-means Clustering Result")
plt.imshow(clustered_image, cmap='tab10')  # 使用'jet'色图来显示聚类结果
plt.axis('off')

plt.show()

# 可选：打印K-means的聚类中心
print("K-means Cluster Centers (Texture Features):")
print(kmeans.cluster_centers_)
