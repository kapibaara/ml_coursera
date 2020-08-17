from skimage.io import imread
import pylab
from skimage.io import imshow
from skimage import img_as_float
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

image = imread('data_seta/parrots.jpg')
pylab.imshow(image)

w, h, d = tuple(image.shape)
X = img_as_float(image).reshape((w * h, d))

kmeans = KMeans(init='k-means++', random_state=241, n_clusters=64)
kmeans.fit(X)

labels = kmeans.labels_


def get_average_color():
    new_image = np.empty((len(labels), 3))
    for i in range(len(labels)):
        label = labels[i]
        new_image[i] = kmeans.cluster_centers_[label]
    return new_image


def get_mean_color():
    cluster_point = [None] * kmeans.n_clusters
    for i in range(len(labels)):
        label = labels[i]
        if cluster_point[label] is None:
            cluster_point[label] = []

        cluster_point[label].append(X[i])

    for i in range(kmeans.n_clusters):
        mean_color_in_cluster = np.mean(cluster_point[i], axis=0)
        cluster_point[i] = mean_color_in_cluster

    new_image_mean = np.empty((len(labels), 3))
    for i in range(len(labels)):
        label = labels[i]
        new_image_mean[i] = cluster_point[label]

    return new_image_mean


def psnr(img1, img2):
    mse = mean_squared_error(img1, img2)
    return 10 * math.log10(1 / mse)


print(psnr(X, get_average_color()))
print(psnr(X, get_mean_color()))

# imshow(get_average_color().reshape((w, h, d)))
# plt.title('average')

imshow(np.reshape(get_mean_color(), (w, h, d)))
plt.title('mean')
plt.show()

