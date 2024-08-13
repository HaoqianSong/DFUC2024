# coding=utf-8
# Partial codes inferenced from TransUNet (https://github.com/Beckschen/TransUNet)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import vit_seg_configs as configs
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import os
import random
import cv2
import scipy.ndimage as ndi
import numpy as np

# CPU加速相关的聚类算法
from sklearn.impute import SimpleImputer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
"""
# GPU加速相关的聚类算法
from cuml.cluster import AgglomerativeClustering
import cuml
import cupy as cp
from cuml.cluster import DBSCAN
"""

torch.set_default_dtype(torch.double)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

class CosineKMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-9):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # 初始化质心
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            # 计算每个样本到质心的余弦距离
            distances = cosine_distances(X, self.centroids)
            # 分配标签
            self.labels_ = np.argmin(distances, axis=1)
            # 计算新的质心
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])
            # 检查是否收敛
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = cosine_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

# KMeans聚类算法  kmeans = CosineKMeans(n_clusters=2)
class CosineKMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-9):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
    def fit(self, X):
        # 初始化质心
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            # 计算每个样本到质心的余弦距离
            distances = cosine_distances(X, self.centroids)
            # 分配标签
            self.labels_ = np.argmin(distances, axis=1)
            # 计算新的质心
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])
            # 检查是否收敛
            if np.all(np.linalg.norm(new_centroids - self.centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids
    def predict(self, X):
        distances = cosine_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

# sklearn中MiniBatchKMeans聚类算法 MiniBatchKMeans 是KMeans的一种变体，适用于大规模数据集。它通过在每次迭代中使用小批量数据来更新簇中心，从而加快计算速度
# kmeans = MiniBatchKMeansCluster(n_clusters=2, max_iter=100)
class MiniBatchKMeansCluster:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.model = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter)
    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.centroids = self.model.cluster_centers_
    def predict(self, X):
        distances = cosine_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

# sklearn中AgglomerativeClustering聚类算法CPU运行  AgglomerativeClustering 是一种层次聚类算法，从每个数据点开始，将最近的两个簇合并，直到所有点都在一个簇中。
# kmeans = AgglomerativeCluster(n_clusters=2)
class AgglomerativeCluster:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = AgglomerativeClustering(n_clusters=n_clusters)
    def fit(self, X):
        self.labels_ = self.model.fit_predict(X)
        unique_labels = np.unique(self.labels_)
        self.centroids = np.array([X[self.labels_ == label].mean(axis=0) for label in unique_labels])
    def predict(self, X):
        distances = cdist(X, self.centroids, 'euclidean')
        return np.argmin(distances, axis=1)
# cuML中AgglomerativeClustering聚类算法GPU运行
# kmeans = GPU_AgglomerativeCluster(n_clusters=2)
class GPU_AgglomerativeCluster:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = AgglomerativeClustering(n_clusters=n_clusters)
    def fit(self, X):
        X_gpu = cp.asarray(X)
        self.labels_ = self.model.fit_predict(X_gpu).get()
        unique_labels = np.unique(self.labels_)
        self.centroids = np.array([X[self.labels_ == label].mean(axis=0) for label in unique_labels])
    def predict(self, X):
        if not hasattr(self, 'centroids') or len(self.centroids) == 0:
            return np.full(X.shape[0], -1)
        distances = cdist(X, self.centroids, 'euclidean')
        return np.argmin(distances, axis=1)

# sklearn中DBSCAN聚类算法CPU运行  DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法，通过找到密度足够高的区域来形成簇
# kmeans = DBSCANCluster(eps=0.5, min_samples=5)
class DBSCANCluster:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
    def fit(self, X):
        self.labels_ = self.model.fit_predict(X)
        unique_labels = np.unique(self.labels_)
        self.centroids = np.array([X[self.labels_ == label].mean(axis=0) for label in unique_labels if label != -1])
    def predict(self, X):
        if not hasattr(self, 'centroids') or len(self.centroids) == 0:
            return np.full(X.shape[0], -1)
        distances = cdist(X, self.centroids, 'euclidean')
        return np.argmin(distances, axis=1)
# cuML中DBSCAN聚类算法GPU运行
# kmeans = DBSCANCluster(eps=0.5, min_samples=5)
class GPU_DBSCANCluster:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
    def fit(self, X):
        X_gpu = cp.asarray(X)
        self.model.fit(X_gpu)
        self.labels_ = self.model.labels_.get()
        unique_labels = np.unique(self.labels_)
        self.centroids = np.array([X[self.labels_ == label].mean(axis=0) for label in unique_labels if label != -1])
    def predict(self, X):
        X_gpu = cp.asarray(X)
        if not hasattr(self, 'centroids') or len(self.centroids) == 0:
            return np.full(X.shape[0], -1)
        distances = cdist(X, self.centroids, 'euclidean')
        return np.argmin(distances, axis=1)

# sklearn中Gaussian Mixture Models (GMM)聚类算法CPU运行   GMM使用高斯分布来建模数据点的分布，通过期望最大化（EM）算法来估计模型参数
#kmeans = GMMCluster(n_clusters=2, max_iter=100)
class GMMCluster:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.model = GaussianMixture(n_components=n_clusters, max_iter=max_iter)
    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        self.centroids = self.model.means_
    def predict(self, X):
        distances = cosine_distances(X, self.centroids)
        return np.argmin(distances, axis=1)

# sklearn中Spectral Clustering聚类算法CPU运行  Spectral Clustering利用图论和谱图理论，通过计算数据点的相似度矩阵并进行特征分解来进行聚类
# kmeans = SpectralCluster(n_clusters=2)
class SpectralCluster:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
    def fit(self, X):
        self.labels_ = self.model.fit_predict(X)
        unique_labels = np.unique(self.labels_)
        self.centroids = np.array([X[self.labels_ == label].mean(axis=0) for label in unique_labels])
    def predict(self, X):
        if not hasattr(self, 'centroids') or len(self.centroids) == 0:
            return np.full(X.shape[0], -1)
        distances = cdist(X, self.centroids, 'euclidean')
        return np.argmin(distances, axis=1)

# sklearn中BIRCH(Balanced Iterative Reducing and Clustering using Hierarchies)聚类算法CPU运行， BIRCH是一种层次聚类算法，适用于大规模数据集，通过构建和调整树结构来进行聚类
# kmeans = BIRCHCluster(n_clusters=2)
class BIRCHCluster:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = Birch(n_clusters=n_clusters)
    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.labels_
        unique_labels = np.unique(self.labels_)
        self.centroids = np.array([X[self.labels_ == label].mean(axis=0) for label in unique_labels])
    def predict(self, X):
        if not hasattr(self, 'centroids') or len(self.centroids) == 0:
            return np.full(X.shape[0], -1)
        distances = cdist(X, self.centroids, 'euclidean')
        return np.argmin(distances, axis=1)

# sklearn中Mean Shift聚类算法CPU运行， Mean Shift 是一种基于密度的聚类算法，通过移动数据点到密度最高的区域来形成簇
# kmeans = MeanShiftCluster(bandwidth=2)
class MeanShiftCluster:
    def __init__(self, bandwidth=2):
        self.bandwidth = bandwidth
        self.model = MeanShift(bandwidth=bandwidth)
    def fit(self, X):
        self.model.fit(X)
        self.labels_ = self.model.labels_
        unique_labels = np.unique(self.labels_)
        self.centroids = np.array([X[self.labels_ == label].mean(axis=0) for label in unique_labels])
    def predict(self, X):
        if not hasattr(self, 'centroids') or len(self.centroids) == 0:
            return np.full(X.shape[0], -1)
        distances = cdist(X, self.centroids, 'euclidean')
        return np.argmin(distances, axis=1)


def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

def extract_and_fill_skin_region(image,output_mask_folder,output_image_folder,name,
        colour_gamut="rgb", num_for=2, num_clusters=2, erode_iterations=30,
        close_iterations=10, open_iterations=2, color_limit=38):
    # 将图像转换为Hsv颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义品红色到红色的Hsv范围
    lower_magenta = np.array([170, color_limit, color_limit], dtype=np.uint8)
    upper_magenta = np.array([179, 255, 255], dtype=np.uint8)

    # 定义红色到黄色的Hsv范围
    lower_yellow = np.array([0, color_limit, color_limit], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    # 基于品红色到红色范围创建掩膜
    mask_magenta = cv2.inRange(hsv, lower_magenta, upper_magenta)

    # 基于红色到黄色范围创建掩膜
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 合并两个掩膜
    mask = cv2.bitwise_or(mask_magenta, mask_yellow)


    # 应用开运算去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    mean_val = cv2.mean(image, mask=mask)[:3]
    # 应用闭运算填充小孔洞

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    # 找到所有联通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 找到最大的联通区域（忽略背景区域）
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # 创建一个新的掩膜，只保留最大的联通区域
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255
    mask = largest_mask
    # 创建一个结构元素
    #kernel = np.ones((3, 3), np.uint8)
    # 对掩膜进行腐蚀操作
    mask = cv2.erode(mask, kernel, iterations=erode_iterations)
    # 应用开运算去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    # 提取mask中的区域
    result = cv2.bitwise_and(image, image, mask=mask)

    # 计算mask区域的平均值
    # 将图像中非mask区域填充为平均值
    #result[mask == 0] = mean_val
    '''# 复制掩膜并填充外围空洞
    mask_filled = mask.copy()
    h, w = mask.shape[:2]
    mask_temp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_filled, mask_temp, (0, 0), 255)'''
    #取反并与原始掩膜结合，得到填充空洞后的掩膜
    #mask_filled = cv2.bitwise_not(mask_filled)
    #mask_filled = cv2.bitwise_or(mask, mask_filled)
    # 提取 mask 区域的 HSV 值
    for i in range(num_for):
        y, x = np.where(mask > 1)
        Lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        Luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
        YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        masked_Lab = Lab[y, x]
        masked_Luv = Luv[y, x]
        masked_YCrCb = YCrCb[y, x]
        masked_YUV = YUV[y, x]
        masked_hsv = hsv[y, x]
        masked_rgb = image[y, x]

        # 归一化 HSV 数据
        masked_hsv = normalize(masked_hsv, norm='l2')
        masked_rgb = normalize(masked_rgb, norm='l2')
        masked_Lab = normalize(masked_Lab, norm='l2')
        masked_Luv = normalize(masked_Luv, norm='l2')
        masked_YCrCb = normalize(masked_YCrCb, norm='l2')
        masked_YUV = normalize(masked_YUV, norm='l2')
        # PCA 降维
        #pca = PCA(n_components=2)
        #reduced_rgb_values = pca.fit_transform(masked_rgb)
        # K-Means 聚类
        kmeans = GMMCluster(n_clusters=num_clusters, max_iter=100)
        #均可用： KMeans(n_clusters=num_clusters) CosineKMeans(n_clusters=num_clusters) MiniBatchKMeansCluster(n_clusters=num_clusters, max_iter=100)
        # AgglomerativeCluster(n_clusters=num_clusters) 内存不足     GPU_AgglomerativeCluster(n_clusters=num_clusters)  安装不上cuml
        # DBSCANCluster(eps=0.5, min_samples=5) 内存不足  GPU_DBSCANCluster(eps=0.5, min_samples=5) 安装不上cuml
        # GMMCluster(n_clusters=num_clusters, max_iter=100)可用  SpectralCluster(n_clusters=num_clusters) 图不连接
        # BIRCHCluster(n_clusters=num_clusters) 子聚类数量不足不分割 MeanShiftCluster(bandwidth=2)
        masked_used = eval(f"masked_{colour_gamut}")
        kmeans.fit(masked_used)
        labels = kmeans.predict(masked_used)
        count = [0] * num_clusters   # 对应n_clusters=2，  例如n_clusters=2时count = [0, 0， 0]
        # 获取分割后的图像
        segmented_mask = np.zeros_like(mask, dtype=np.uint8)
        choice = []
        for j, (yy, xx) in enumerate(zip(y, x)):
            segmented_mask[yy, xx] = labels[j] + 1
        for k in range(num_clusters):   # 对应n_clusters=2，   例如n_clusters=3时for k in range(3):
            mask_part = (segmented_mask==k+1).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            #kernel = np.ones((3, 3), np.uint8)
            #largest_component_mask = cv2.morphologyEx(largest_component_mask, cv2.MORPH_OPEN, kernel, iterations=10)
            mask_part = cv2.morphologyEx(mask_part, cv2.MORPH_CLOSE, kernel, iterations=close_iterations+15)
            # Fill holes
            filled_mask = ndi.binary_fill_holes(mask_part).astype(np.uint8)

            count[k] = np.sum(filled_mask)
            choice.append(filled_mask)
        # Find the area of each segmented part
        #unique_labels, counts = np.unique(labels, return_counts=True)
        '''if len(counts) < 2:
            return  # Not enough segments to process'''

        # Find the label of the smaller part
        small_label = np.argmin(count)

        # Get the mask of the smaller part
        '''small_part_mask = np.zeros_like(segmented_mask, dtype=np.uint8)
        small_part_mask[segmented_mask == small_label+1] = 255'''
        small_part_mask=choice[small_label]

        # Find the largest connected component
        labeled_array, num_features = ndi.label(small_part_mask)
        sizes = ndi.sum(small_part_mask, labeled_array, range(num_features + 1))
        largest_component = sizes.argmax()
        largest_component_mask = (labeled_array == largest_component).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        #kernel = np.ones((3, 3), np.uint8)
        #largest_component_mask = cv2.morphologyEx(largest_component_mask, cv2.MORPH_OPEN, kernel, iterations=10)
        largest_component_mask = cv2.morphologyEx(largest_component_mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
        # Fill holes
        filled_mask = ndi.binary_fill_holes(largest_component_mask).astype(np.uint8) * 255
        mask = filled_mask

    # 创建一个浅色填充的图像
    light_fill_color = (0, 0, 255)
    filled_image = image.copy()
    filled_image[mask == 0] = light_fill_color
    alpha = 0.8  # 透明度
    # Save the results
    mask_output_path = os.path.join(output_mask_folder, name)
    image_output_path = os.path.join(output_image_folder, name)
    image_mask_output_path = os.path.join(output_image_folder + "_mask", name)

    cv2.imwrite(mask_output_path, mask)
    cv2.imwrite(image_output_path, cv2.bitwise_and(image, image, mask=mask))
    cv2.imwrite(image_mask_output_path, cv2.addWeighted(image, alpha, filled_image, 1 - alpha, 0))
    return result

def save_folder(colour_gamut = "rgb", num_for = 2, num_clusters = 2, erode_iterations = 30, close_iterations = 10, open_iterations = 2, color_limit = 38):
    output_folder1 = f"/home/pcl/sdb1/data/stal_pakg/Medical_Segmentation/fanDFUCout/Guo_cutler_output6/SkinPreFH_Open{open_iterations}Close{close_iterations}Erode{erode_iterations}GMMCluster{num_clusters}FOR{num_for}{colour_gamut}Color{color_limit}/output_img"  # ./SkinPre_Kmeans/output_img  /raid/wangyifan/Explicit/Explicit-Shape-Priors-main/output_img
    output_folder2 = f"/home/pcl/sdb1/data/stal_pakg/Medical_Segmentation/fanDFUCout/Guo_cutler_output6/SkinPreFH_Open{open_iterations}Close{close_iterations}Erode{erode_iterations}GMMCluster{num_clusters}FOR{num_for}{colour_gamut}Color{color_limit}/output_label"  # ./SkinPre_Kmeans/output_label  /raid/wangyifan/Explicit/Explicit-Shape-Priors-main/output_label
    output_folder3 = output_folder1 + "_mask"

    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)
    if not os.path.exists(output_folder3):
        os.makedirs(output_folder3)
    return output_folder1, output_folder2


if __name__ == "__main__":
    color_limit_list = [18, 28, 38, 48]  #38#  28, 33, 38, 43, 48
    open_iterations_list = [1, 2, 3]  #2#
    close_iterations_list = [23, 25, 27]   #原10但25时效果最好#   5, 10, 20, 25, 30
    erode_iterations_list = [10, 20, 30]  #10#, 40
    num_clusters_list = [3]  #2#2,
    num_for_list = [2]   #2#1, 2, 3 3,4
    colour_gamut_list = ["rgb"]  # 已经确定"rgb"最好： "hsv", "rgb", "Lab", "Luv", "YCrCb", "YUV"
    for num_for in num_for_list:
        for num_clusters in num_clusters_list:
            for erode_iterations in erode_iterations_list:
                for close_iterations in close_iterations_list:
                    for open_iterations in open_iterations_list:
                        for color_limit in color_limit_list:
                            input_folder = './DFUCdata/val_cutler_output'  #val_cutler_output DFUC2024_test_release /raid/wangyifan/Explicit/Explicit-Shape-Priors-main/data_DFU
                            output_folder1, output_folder2 = save_folder(color_limit=color_limit, open_iterations=open_iterations,
                                close_iterations=close_iterations, erode_iterations=erode_iterations,num_clusters=num_clusters,num_for=num_for)
                            print(output_folder1) #

                            # 获取输入文件夹中的所有图片文件
                            image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
                            for image_file in image_files:
                                # 读取图像
                                image_path = os.path.join(input_folder, image_file)
                                print(image_path)
                                image = cv2.imread(image_path)

                                if image is None:
                                    print(f"Failed to read image {image_path}")
                                    continue

                                # 提取并填充肤色区域
                                mask_filled = extract_and_fill_skin_region(image,output_folder2,output_folder1,image_file,
                                    color_limit=color_limit,open_iterations=open_iterations,close_iterations=close_iterations,
                                    erode_iterations=erode_iterations,num_clusters=num_clusters,num_for=num_for)


                                # 保存处理后的掩膜
                                #output_path = os.path.join(output_folder1, image_file)
                                #cv2.imwrite(output_path, mask_filled)

