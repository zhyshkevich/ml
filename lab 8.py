import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math


# 1. Загрузите данные ex8data1.mat из файла.

data1 = scipy.io.loadmat('ex8data1.mat')
X1 = data1['X']
X1_val = data1['Xval']
y1_val = data1['yval']
print(X1.shape)

# 2. Постройте график загруженных данных в виде диаграммы рассеяния.

def draw_scatter(X):
    plt.figure(figsize=(8,8))
    plt.scatter(X[:,0], X[:,1], s=9, marker='.')
    plt.xlabel('Задержка (мс)',fontsize=14)
    plt.ylabel('Пропускная способность (мб/с)',fontsize=14)
    plt.grid(True)
    plt.show()

draw_scatter

# 3. Представьте данные в виде двух независимых нормально распределенных случайных величин.

def draw_histogram(X, title=None):
    plt.figure(figsize=(8,5))
    plt.hist(X, 100)
    if title:
        plt.title(title, fontsize=20)
    plt.grid(True)
    plt.show()

draw_histogram(X1[:, 0], 'Задержка (мс)')

draw_histogram(X1[:,1], 'Пропускная способность (мб/с)')

# 4. Оцените параметры распределений случайных величин.

def get_dist_params(X):
    return np.mean(X, axis=0), np.var(X, axis=0)

mu, sig2 = get_dist_params(X1)

print(f'\u03BC = {mu}')
print(f'\u03C3^2 = {sig2}')

# 5. Постройте график плотности распределения получившейся случайной величины в виде изолиний, совместив его с
# графиком из пункта 2.

def p(X, mu, sigma2):
    p1 = 1 / (np.sqrt(2 * math.pi) * np.sqrt(sigma2))
    p2 = ((X - mu) ** 2) / (2 * sigma2)
    result_matrix = p1 * np.exp(-p2)

    return np.prod(result_matrix, axis=1).reshape(-1, 1)


def draw_dist(mu, sigma2):
    grid_params = np.arange(0, 29, 0.2)
    x1, x2 = np.meshgrid(grid_params, grid_params)
    z = np.column_stack([x1.flatten(), x2.flatten()])
    z = p(z, mu, sigma2)
    z = z.reshape(x1.shape)
    levels = [10 ** exp for exp in range(-20, 0, 3)]
    plt.contour(x1, x2, z, levels=levels)
    plt.show()


draw_scatter(X1)
draw_dist(mu, sig2)

# 6. Подберите значение порога для обнаружения аномалий на основе валидационной выборки. В качестве метрики
# используйте F1-меру.

def f1_score(y_true, y_pred):
    # print(f'y_true = {y_true.shape}, y_pred = {y_pred.shape}')
    assert y_true.shape == y_pred.shape

    tp = np.sum(np.logical_and((y_true == 1), (y_pred == 1)))
    fp = np.sum(np.logical_and((y_true == 0), (y_pred == 1)))
    fn = np.sum(np.logical_and((y_true == 1), (y_pred == 0)))

    if (tp + fp) == 0 or (tp + fn) == 0: return 0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    result = 2 * (precision * recall) / (precision + recall)

    if math.isnan(result):
        result = 0

    return result


def find_eps(y_true, p_vals, iterations=100):
    epsilons = np.linspace(np.max(p_vals), np.min(p_vals), iterations)

    best_f1 = 0
    best_eps = 1
    best_iteration = 0

    for i in range(len(epsilons)):
        eps = epsilons[i]
        y_pred = p_vals < eps
        f1 = float(f1_score(y_true, y_pred))
        if f1 > best_f1:
            best_f1 = f1
            best_eps = eps
            best_iteration = i

    return best_eps, best_f1, best_iteration


p_vals = p(X1_val, mu, sig2)
best_eps, best_f1, best_i = find_eps(y1_val, p_vals, 200)

print(f'best_eps: {best_eps}')
print(f'best_f1: {best_f1}')
print(f'best_iteration: {best_i}')

# 7. Выделите аномальные наблюдения на графике из пункта 5 с учетом выбранного порогового значения.


def highlite_anomalies(X, mu, sig2, eps):
    p_vals = p(X, mu, sig2)
    indices = np.nonzero(p_vals < eps)[0]

    anomalies = X[indices]
    plt.scatter(anomalies[:, 0], anomalies[:, 1], s=150, facecolors='none', edgecolors='r')
    plt.show()

draw_scatter(X1)
draw_dist(mu, sig2)
highlite_anomalies(X1, mu, sig2, best_eps)

# 8. Загрузите данные ex8data2.mat из файла.

data2 = scipy.io.loadmat('ex8data2.mat')

X2 = data2['X']
X2_val = data2['Xval']
y2_val = data2['yval']
X2.shape

# 9. Представьте данные в виде 11-мерной нормально распределенной случайной величины.

fig1, axes = plt.subplots(ncols=6, nrows=2, constrained_layout=True, figsize=(14, 4))
m, n = X2.shape

axes = axes.flatten()

for i in range(len(axes)):
    ax = axes[i]

    if i >= n:
        ax.axis('off')
        continue

    ax.hist(X2[:, i], 100)
    ax.set_title(f'X{i}')
    ax.grid(True)

# 10. Оцените параметры распределения случайной величины.

def get_dist_params_multi(X):
    mu = np.mean(X, axis=0)
    sig_p = X - mu
    Sigma = np.dot(sig_p.T, sig_p) / len(X)

    return mu, Sigma

import pandas as pd

mu_2, Sigma_2 = get_dist_params_multi(X2)

data = np.column_stack((mu_2, Sigma_2))
df = pd.DataFrame(data)
df.rename(columns={0: '\u03BC', 1: '\u03C3^2'}, inplace=True)
print(df)

# 11. Подберите значение порога для обнаружения аномалий на основе валидационной выборки.
# В качестве метрики используйте F1-меру.

def p_multi(X, mu, Sigma):
    m, n = X.shape

    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.pinv(Sigma)

    e1 = 1 / (np.power((2 * math.pi), n / 2) * np.sqrt(Sigma_det))

    X_mu = X - mu

    e2 = np.exp(- 0.5 * np.sum((np.dot(X_mu, Sigma_inv) * X_mu), axis=1))

    return (e1 * e2).reshape(-1, 1)

p_vals_2 = p_multi(X2_val, mu_2, Sigma_2)
best_eps_2, best_f1_2, best_i_2 = find_eps(y2_val, p_vals_2, 6000)

print(f'best_eps: {best_eps_2}')
print(f'best_f1: {best_f1_2}')
print(f'best_iteration: {best_i_2}')

# 12. Выделите аномальные наблюдения в обучающей выборке. Сколько их было обнаружено? Какой был подобран порог?

p_vals_2_train = p_multi(X2, mu_2, Sigma_2)
indices = np.nonzero(p_vals_2_train < best_eps_2)[0]
print(f'Найдено "аномалии" на обучающей выборке: {len(indices)}')








