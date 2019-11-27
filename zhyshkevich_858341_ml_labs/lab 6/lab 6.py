import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import random
from mlxtend.plotting import category_scatter
import imageio
from sklearn.cluster import AgglomerativeClustering


# 1. Загрузите данные ex6data1.mat из файла.

data1 = scipy.io.loadmat('ex6data1.mat')
X1 = data1['X']
print(X1.shape)

fig = plt.figure(figsize=(8,5))
plt.scatter(X1[:, 0], X1[:, 1])
plt.show()


# 2. Реализуйте функцию случайной инициализации K центров кластеров.


def rand_init_centroids(X, K):
    indexes = random.sample(range(0, len(X)), K)

    return X[indexes]

print(rand_init_centroids(X1, 3))


# 3. Реализуйте функцию определения принадлежности к кластерам.

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum(np.square((p1 - p2))))

print(euclidean_distance(np.array([2, 1]), np.array([2, 3])))


def assign_clusters(X, centroids):
    m = len(X)

    c = np.zeros([m, 1]).astype(int)

    for x_i in range(m):
        x = X[x_i]
        x_distance = 100000  # инициализируем большим занчением

        for c_i in range(len(centroids)):
            centroid = centroids[c_i]
            dist = euclidean_distance(x, centroid)

            if dist < x_distance:
                x_distance = dist
                c[x_i] = int(c_i)

    return c


c = assign_clusters(X1, rand_init_centroids(X1, 2))
print(c.shape)

# 4. Реализуйте функцию пересчета центров кластеров.

def split_data_by_clusters(X, clusters):
    m = len(X)
    cluster_indexes = np.unique(clusters)

    cluster_values = []

    for c_index in cluster_indexes:
        # получаем все значения для центроида c_index
        values = np.array([X[i] for i in range(m) if clusters[i] == c_index])
        cluster_values.append(values)

    return cluster_values

def move_centroids(X, clusters):
    cluster_valaues = split_data_by_clusters(X, clusters)
    return np.array([np.mean(c_vals, axis=0) for c_vals in cluster_valaues])

print(move_centroids(X1, c))


# 5. Реализуйте алгоритм K-средних.

def k_means(X, K, max_iter=10):
    centroids = rand_init_centroids(X, K)
    centroids_history = [centroids]
    clusters = np.zeros([len(X), 1])

    for i in range(max_iter):
        clusters = assign_clusters(X, centroids)
        centroids = move_centroids(X, clusters)
        centroids_history.append(centroids)

    return centroids, clusters, np.array(centroids_history)


def k_mean_cost(X, clusters, centroids):
    m = len(clusters)

    sum = 0
    for i in range(len(X)):
        sum += euclidean_distance(X[i], centroids[int(clusters[i])])

    return sum / m


def best_k_means(X, K, max_iter=10, tries=100):
    best_cost = 1000000
    best_centr = None
    best_cluster = None
    best_centr_hist = None
    best_iteration = 0

    for i in range(tries):
        centr, clust, centr_hist = k_means(X, K, max_iter)
        cost = k_mean_cost(X, clust, centr)
        if cost < best_cost:
            best_cost = cost
            best_centr = centr
            best_cluster = clust
            best_centr_hist = centr_hist
            best_iteration = i

    print(f'Лучший результат: {best_cost} был достигнут на {best_iteration} итерации.')

    return best_centr, best_cluster, best_centr_hist


# 6. Постройте график, на котором данные разделены на K=3 кластеров (при помощи различных маркеров или цветов),
# а также траекторию движения центров кластеров в процессе работы алгоритма

centr, clust, centr_hist = best_k_means(X1, 3, 20)
X1_mod = np.hstack([X1, clust])

colors = ['blue', 'green','purple', 'gray', 'cyan']
fig = category_scatter(x=0, y=1, label_col=2, data=X1_mod, markersize=35, colors=colors, markers='so^v')
fig.set_size_inches(10, 5)

for i in range(len(centr)):
    cntrid = centr[i]
    plt.plot(cntrid[0], cntrid[1], 'x', markersize=35, c=colors[i])
    plt.plot(centr_hist[:,i,0], centr_hist[:,i,1], 'rx--', markersize=5)

plt.show()


# 7. Загрузите данные bird_small.mat из файла.

bird_data = scipy.io.loadmat('bird_small.mat')
Xb = bird_data['A']
print(Xb.shape)

plt.imshow(Xb)
plt.show()

# 8. С помощью алгоритма K-средних используйте 16 цветов для кодирования пикселей.

bird_k = 16

bird_centr, bird_clust, _ = best_k_means(Xb.reshape(-1, 3), bird_k, 20, 10)


# 9. Насколько уменьшился размер изображения? Как это сказалось на качестве?

xb_size = Xb.size
compressed_size = bird_centr.size + bird_clust.size
size_diff = compressed_size / xb_size

print(f'Размер исходного изображения = {Xb.shape[0]} x {Xb.shape[1]} x {Xb.shape[2]} = {xb_size}')
print(f'Размер сжатого изображения = {bird_centr.shape[0]} x {bird_centr.shape[1]} + {bird_clust.shape[0]} = {compressed_size}')
print(f'Разница = {size_diff:f}')


def reconstruct_image(clusters, centroids):
    pixels_count = clusters.shape[0]
    image = np.zeros([pixels_count, centroids.shape[1]]).astype(int)

    for i in range(pixels_count):
        image[i] = centroids[int(clusters[i])]

    return image

restored_bird_img = reconstruct_image(bird_clust, bird_centr).reshape(128, 128, 3)

print(restored_bird_img.shape)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_title('Оригинал')
ax1.imshow(Xb)
ax2.set_title('Сжатое')
ax2.imshow(restored_bird_img)

plt.show()

img2 = imageio.imread('225-128x128.jpg')
print(img2.shape)

img2_centroids, img2_clusters, _ = best_k_means(img2.reshape(-1, 3), bird_k, 5, 5)
restored_img2 = reconstruct_image(img2_clusters, img2_centroids).reshape(128, 128, 3)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.set_title('Оригинал')
ax1.imshow(img2)
ax2.set_title('Сжатое')
ax2.imshow(restored_img2)
plt.show()

# 11. Реализуйте алгоритм иерархической кластеризации на том же изображении. Сравните полученные результаты.

cluster = AgglomerativeClustering(n_clusters=bird_k)
cluster.fit_predict(img2.reshape(-1, 3))

img2_2_clusters = cluster.labels_
img2_2_centroids = move_centroids(img2.reshape(-1, 3), img2_2_clusters)

restored_img3 = reconstruct_image(img2_2_clusters, img2_2_centroids).reshape(128, 128, 3)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
f.set_size_inches(10, 9)
ax1.set_title('Оригинал')
ax1.imshow(img2)
ax2.set_title('Иерархический')
ax2.imshow(restored_img3)
ax3.set_title('К-средних')
ax3.imshow(restored_img2)
plt.show()


