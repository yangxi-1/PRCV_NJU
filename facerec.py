import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载数据


def load_data(data_dir):
    images = []
    labels = []
    # 遍历人脸数据集中的每个人的文件夹，list_dir()返回指定目录下的文件和文件夹列表
    for person_dir in os.listdir(data_dir):
        if not person_dir.startswith('s'):
            continue
        person_id = int(person_dir.replace('s', ''))
        label = person_id-1
        person_path = os.path.join(data_dir, person_dir)

        for filename in os.listdir(person_path):
            if not filename.endwith('.pgm'):
                continue
            image_path = os.path.join(person_path, filename)
            # 以灰度图像读入,pgm文件是灰度图像
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            img = img.flatten()
            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# 划分数据集


def split_data(images, labels):
    n_samples, n_features = images.shape
    n_train = int(0.7*n_samples)  # 70%作为训练集
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # 按类别划分数据集(分层抽样)
    for label in np.unique(labels):
        idx = np.where(train_labels == label)[0]
        train_idx = idx[:int(0.8*len(idx))]
        test_idx = idx[int(0.8*len(idx)):]
        train_images.append(images[train_idx])
        train_labels.append(labels[train_idx])
        test_images.append(images[test_idx])
        test_labels.append(labels[test_idx])

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    return train_images, train_labels, test_images, test_labels


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)  # axis=0表示按列求均值
        Xc = X-self.mean

        # 计算协方差矩阵,Xc.shape[0]-1是为了得到无偏估计
        cov_matrix = Xc.T.dot(Xc)/(Xc.shape[0]-1)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # 计算特征值和特征向量

        # 对特征向量排序
        # argsort()返回的是数组值从小到大的索引值，[::-1]两个冒号表示逆序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 转换特征向量到原始空间
        self.components = Xc.T@eigenvectors[:, :self.n_components]
        self.components = self.components / \
            np.linalg.norm(self.components, axis=0)  # 归一化
        return self


data_dir = 'att_faces'
images, labels = load_data(data_dir)
images = images/255.0

train_images, train_labels, test_images, test_labels = split_data(
    images, labels)

n_components = 100
pca = PCA(n_components=n_components)

pca.fit(train_images)

train_images_pca = pca.transform(train_images)
test_images_pca = pca.transform(test_images)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_images_pca, train_labels)
pred_labels = knn.predict(test_images_pca)
accuracy = accuracy_score(test_labels, pred_labels)
print('Accuracy of PCA:', accuracy)


# 可视化前几个主成分（可选）

plt.figure(figsize=(10, 4))
for i in range(6):
    plt.subplot(2, 3, i+1)
    eigenface = pca.components[:, i].reshape(112, 92)  # 假设图像尺寸为112x92（ORL标准）
    plt.imshow(eigenface, cmap='gray')
    plt.title(f"PC {i+1}")
    plt.axis('off')
plt.suptitle("First 6 Principal Components (Eigenfaces)")
plt.show()
