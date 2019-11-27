import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions

# 1. Загрузите данные ex5data1.mat из файла.


data = scipy.io.loadmat('ex5data1.mat')

X = data['X']
y = data['y']

# 2. Постройте график для загруженного набора данных: по осям - переменные X1, X2, а точки, принадлежащие различным
# классам должны быть обозначены различными маркерами.

def draw_data(X, y):
    pos = X[np.where(y==1)[0]]
    neg = X[np.where(y==0)[0]]

    plt.figure(figsize=(8,5))
    plt.scatter(pos[:,0], pos[:,1], marker='o', facecolors='none', edgecolors='b')
    plt.scatter(neg[:,0], neg[:,1], c='r', marker='x')
    plt.show()

draw_data(X, y)

# 3. Обучите классификатор с помощью библиотечной реализации SVM с линейным ядром на данном наборе.

linear_svm_c1 = svm.SVC(C=1, kernel='linear')
linear_svm_c1.fit( X, y.flatten())


# 4. Постройте разделяющую прямую для классификаторов с различными параметрами C = 1, C = 100 (совместно с графиком
# из пункта 2). Объясните различия в полученных прямых?

def draw_decision_boundry(svm, X, y):
    fig = plt.figure(figsize=(8,5))
    ax = plot_decision_regions(X, y.flatten(), clf=svm, markers='xo', colors='r,b', hide_spines=False)
    plt.show()

draw_decision_boundry(linear_svm_c1, X, y)

linear_svm_c100 = svm.SVC(C=100, kernel='linear')
linear_svm_c100.fit( X, y.flatten() )

draw_decision_boundry(linear_svm_c100, X, y)

# При C = 1 алгоритм попытался поулчить наибольший "margin" между классами, при этом некоторые "выбросы" были
# классифицированы ошибочно, но общий паттерн алгоритм определил. При C = 100 видно, что верно были классифицированы
# все данные, но "margin" сильно уменьшился. Видно, что модель страдает от переобучения (overfitting high variance)

# 5. Реализуйте функцию вычисления Гауссового ядра для алгоритма SVM.


def gaussian_kernel_similarity(x1, l1, sigma):
    diff = x1 - l1
    numerator = np.dot(diff.T, diff)  # нужно взять корень квадратный а потом возьвести в 2 степень
    denominator = (2 * (sigma ** 2))
    return np.exp(-numerator / denominator)

print(gaussian_kernel_similarity(np.array([3, 3, 3]),np.array([1, 1, 1]), 2))


def create_gaussian_f(X, L, sigma):
    xm = len(X)
    lm = len(L)
    fs = np.zeros([xm, lm])
    for xi in range(xm):
        x = X[xi]
        for lj in range(lm):
            l = L[lj]
            fs[xi][lj] = gaussian_kernel_similarity(x, l, sigma)

    return fs


# 6. Загрузите данные ex5data2.mat из файла.

data2 = scipy.io.loadmat('ex5data2.mat')

X2 = data2['X']
y2 = data2['y']

draw_data(X2, y2)

sigma_ex2 = 0.1

F = create_gaussian_f(X2, X2, sigma_ex2)
print(F.shape)

svm_ex_2 = svm.SVC(C=1, kernel='linear')
svm_ex_2.fit( F, y2.flatten())


def visualize_boundary(L, y, svm, sigma):
    draw_data(L, y)

    x1_greed = np.linspace(L[:, 0].min(), L[:, 0].max(), 100)
    x2_greed = np.linspace(L[:, 1].min(), L[:, 1].max(), 100)

    X1, X2 = np.meshgrid(x1_greed, x2_greed)
    Z = np.zeros(X1.shape)

    for i in range(X1.shape[1]):
        x = np.column_stack((X1[:, i], X2[:, i]))
        f = create_gaussian_f(x, L, sigma)
        Z[:, i] = svm.predict(f)

    plt.contour(X1, X2, Z, colors="black", levels=[0, 0])
    plt.show(block=False)


visualize_boundary(X2, y2, svm_ex_2, sigma_ex2)

def sigma_to_gamma_ang(sigma):
    return np.power(sigma, -2.) / 2

print(sigma_to_gamma_ang(2))


svm_ex_2_alt = svm.SVC(C=1, kernel='rbf', gamma=sigma_to_gamma_ang(sigma_ex2))
svm_ex_2_alt.fit( X2, y2.flatten() )


fig = plt.figure(figsize=(10, 6))
ax = plot_decision_regions(X2, y2.flatten(), clf=svm_ex_2_alt, markers='xo', colors='r,b', zoom_factor=8)
plt.show()

# Набор данных ex5data3.mat представляет собой файл формата *.mat (т.е. сохраненного из Matlab). Набор содержит три
# переменные X1 и X2 (независимые переменные) и y (метка класса). Данные разделены на две выборки: обучающая выборка
# (X, y), по которой определяются параметры модели; валидационная выборка (Xval, yval), на которой настраивается
# коэффициент регуляризации и параметры Гауссового ядра.


# 10. Загрузите данные ex5data3.mat из файла.

data3 = scipy.io.loadmat('ex5data3.mat')

X3_train = data3['X']
y3_train = data3['y']
X3_val = data3['Xval']
y3_val = data3['yval']

draw_data(X3_train, y3_train)

# 11. Вычислите параметры классификатора SVM на обучающей выборке, а также подберите параметры C и σ2 на
# валидационной выборке.

C_vals = [0.01, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10, 20, 40, 80, 100]
sigma_vals = [0.01, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.4, 0.8, 1, 2, 5, 10, 20, 40, 80, 100]

best_score = 0
params = {
    'C': 0,
    'sigma': 0
}

for C in C_vals:
    for sigma in sigma_vals:
        model = svm.SVC(C=C, kernel='rbf', gamma=sigma_to_gamma_ang(sigma))
        model.fit(X3_train, y3_train.flatten())
        score = model.score(X3_val, y3_val)
        if score > best_score:
            best_score = score
            params['C'] = C
            params['sigma'] = sigma

print(f"Лучшая комбинация набрала {best_score}: С = {params['C']}, sigma = {params['sigma']}")


# 12. Визуализируйте данные вместе с разделяющей кривой (аналогично пункту 4).

best_model = svm.SVC(C=params['C'], kernel='rbf', gamma=sigma_to_gamma_ang(params['sigma']))
best_model.fit(X3_train, y3_train.flatten())

fig = plt.figure(figsize=(10, 6))
ax = plot_decision_regions(X3_train, y3_train.flatten(), clf=best_model, markers='xo', colors='r,b', zoom_factor=8)
plt.show()