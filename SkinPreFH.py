# coding=utf-8
# Partial codes inferenced from TransUNet (https://github.com/Beckschen/TransUNet)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.preprocessing import normalize
import os
import random
import cv2
import scipy.ndimage as ndi
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.mixture import GaussianMixture


torch.set_default_dtype(torch.double)


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

    # 对掩膜进行腐蚀操作
    mask = cv2.erode(mask, kernel, iterations=erode_iterations)
    # 应用开运算去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    # 提取mask中的区域
    result = cv2.bitwise_and(image, image, mask=mask)

    # 提取 mask 区域的 HSV 值
    for i in range(num_for):
        y, x = np.where(mask > 1)
        masked_rgb = image[y, x]

        # 归一化 HSV 数据
        masked_rgb = normalize(masked_rgb, norm='l2')
        # K-Means 聚类
        kmeans = GMMCluster(n_clusters=num_clusters, max_iter=100)
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
            mask_part = cv2.morphologyEx(mask_part, cv2.MORPH_CLOSE, kernel, iterations=close_iterations+15)
            # Fill holes
            filled_mask = ndi.binary_fill_holes(mask_part).astype(np.uint8)

            count[k] = np.sum(filled_mask)
            choice.append(filled_mask)

        # Find the label of the smaller part
        small_label = np.argmin(count)

        # Get the mask of the smaller part
        small_part_mask=choice[small_label]

        # Find the largest connected component
        labeled_array, num_features = ndi.label(small_part_mask)
        sizes = ndi.sum(small_part_mask, labeled_array, range(num_features + 1))
        largest_component = sizes.argmax()
        largest_component_mask = (labeled_array == largest_component).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
    output_folder1 = f"/home/pcl/sdb1/data/stal_pakg/Medical_Segmentation/fanDFUCout/Guo_cutler1/SkinPreFH_Open{open_iterations}Close{close_iterations}Erode{erode_iterations}GMMCluster{num_clusters}FOR{num_for}{colour_gamut}Color{color_limit}/output_img"  # ./SkinPre_Kmeans/output_img  /raid/wangyifan/Explicit/Explicit-Shape-Priors-main/output_img
    output_folder2 = f"/home/pcl/sdb1/data/stal_pakg/Medical_Segmentation/fanDFUCout/Guo_cutler1/SkinPreFH_Open{open_iterations}Close{close_iterations}Erode{erode_iterations}GMMCluster{num_clusters}FOR{num_for}{colour_gamut}Color{color_limit}/output_label"  # ./SkinPre_Kmeans/output_label  /raid/wangyifan/Explicit/Explicit-Shape-Priors-main/output_label
    output_folder3 = output_folder1 + "_mask"

    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)
    if not os.path.exists(output_folder3):
        os.makedirs(output_folder3)
    return output_folder1, output_folder2


if __name__ == "__main__":
    color_limit = 38
    open_iterations = 2
    close_iterations = 25
    erode_iterations = 10
    num_clusters = 2
    num_for = 3
    input_folder = './DFUCdata/text_out1'
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




