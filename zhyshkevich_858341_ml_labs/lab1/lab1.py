import pandas
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

# 1 Загрузите набор данных ex1data1.txt из текстового файла.

dataFrame = pandas.read_csv("ex1data1.txt", header=None, names=["population", "incomes"])
print(dataFrame.head())

population = dataFrame["population"].values
incomes = dataFrame["incomes"].values

# 2 Постройте график зависимости прибыли ресторана от населения города, в котором он расположен.

dataFrame.plot(kind='scatter', x='population', y='incomes', color='red')
plt.show()


# 3 Реализуйте функцию потерь J(θ) для набора данных ex1data1.txt.

def calcHypotesis(theta,  x):
    return theta[0] + theta[1] * x


def costFunction(theta, x, y):
    return sum((calcHypotesis(theta, x_i) - y_i) ** 2 for x_i, y_i in zip(x, y)) / len(x) * 2

# 4 Реализуйте функцию градиентного спуска для выбора параметров модели.
# Постройте полученную модель (функцию) совместно с графиком из пункта 2.


def derivative_theta0(theta, x, y, alpha):
    return alpha * sum(calcHypotesis(theta, x_i) - y_i for x_i, y_i in zip(x, y)) / len(x)


def derivative_theta1(theta, x, y, alpha):
    return alpha * sum((calcHypotesis(theta, x_i) - y_i) * x_i for x_i, y_i in zip(x, y)) / len(x)


def gradient_descent (X, Y, iterations = 400, alpha = 0.01 , theta = [0, 0]):
    i = 0
    history = []

    cost = costFunction(theta, X, Y)
    history.append(np.array([cost, np.array(theta)]))

    for it_number in range(iterations):
        tmptheta = theta
        tmptheta[0] = theta[0] - derivative_theta0(theta, X, Y, alpha)
        tmptheta[1] = theta[1] - derivative_theta1(theta, X, Y, alpha)

        cost = costFunction(theta, X, Y)
        history.append(np.array([cost, np.array(theta)]))

    return np.array(theta), cost, np.array(history)


theta, cost, history = gradient_descent(population, incomes, 2000, 0.01, [0, 0])

print(f'Cost: {cost}')
print(f'Theta: {theta}')

cost_history = history[:, 0]
cost_history = np.delete(cost_history, [0, 1])

plt.figure(figsize=(10, 6))
plt.plot(range(cost_history.size), cost_history, 'o-')
plt.grid(True)
plt.xlabel("Iterations")
plt.ylabel("Cost function values")
plt.show()

pred_incomes = calcHypotesis(theta, population)

plt.figure(figsize=(10,6))
plt.plot(population, incomes, '.', color='red')
plt.plot(population, pred_incomes, 'b-', color='green', label = 'Hypotesis: h(x) = %0.2f + %0.2fx'%(theta[0],theta[1]))
plt.grid(True)
plt.ylabel('Incomes')
plt.xlabel('Population')
plt.legend()
plt.show()

# 5 Постройте трехмерный график зависимости функции потерь от параметров модели (θ0 и θ1) как в виде поверхности,
# так и в виде изолиний (contour plot).

theta_ = theta.reshape(-1, 1)

x_3d = np.arange(-4.2, 1, 0.02)
y_3d = np.arange(-0.1, 1.5, 0.02)

plot_x, plot_y = np.meshgrid(x_3d, y_3d)
plot_z = np.array([costFunction(np.array([plot_x[i], plot_y[i]]), population, incomes) for i in range(len(plot_x))])

# выбираем каждый 20 элемент после 5ого
tmp_his = np.concatenate([history[0:5], history[5::20]])

graph_cost_history = np.vstack(tmp_his[:, 0]).astype(np.float).reshape(-1)

tmp = np.vstack(tmp_his[:, 1]).astype(np.float)

theha_0_history = tmp[:, 0].reshape(-1)
theha_1_history = tmp[:, 1].reshape(-1)

fig = plt.figure(figsize=(12,12))
ax = plt.axes(projection='3d')


ax.plot_surface(plot_x, plot_y, plot_z,  rstride=1, cstride=1, linewidth=0, cmap = 'jet', edgecolor='none', alpha=.9)
ax.plot(*theta_, np.array([cost]), 'r*', markersize=14, zorder = 10)
ax.plot(theha_0_history, theha_1_history, graph_cost_history, marker = '.', color = 'r', alpha = .4, zorder = 9)

ax.set_xlabel(r'$\theta_0$', fontsize=18)
ax.set_ylabel(r'$\theta_1$', fontsize=18)
ax.set_zlabel('Cost func', fontsize=18)
ax.view_init(25, 65)

x_3d = np.arange(-4.2, 1, 0.02)
y_3d = np.arange(-0.1, 1.5, 0.02)

plot_x, plot_y = np.meshgrid(x_3d, y_3d)

plot_z = np.array(list(costFunction(np.array([plot_x[i], plot_y[i]]), population, incomes) for i in range(len(plot_x))))
plt.show()

fig = plt.figure(figsize=(10,6))
ax = plt.axes()
cnt = ax.contour(plot_x, plot_y, plot_z, 100, cmap='jet')

#Angles needed for quiver plot
anglesx = theha_0_history[1:] - theha_0_history[:-1]
anglesy = theha_1_history[1:] - theha_1_history[:-1]

ax.quiver(theha_0_history[:-1], theha_1_history[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .9)

plt.show()

# 6 Загрузите набор данных ex1data2.txt из текстового файла.

original_df = pandas.read_csv('ex1data2.txt', header=None, names=['area', 'rooms', 'price'])
print(original_df.head())

Y = original_df['price'].values.astype('float64')
print(Y.shape)

# 7 Произведите нормализацию признаков. Повлияло ли это на скорость сходимости градиентного спуска?
# Ответ дайте в виде графика.

def gd_old(X, Y, iterations=200, alpha=0.01):
    features_count = X.shape[1]

    theta = np.zeros(features_count)

    def h(theta, x):
        sum = 0
        for i in range(len(theta)):
            sum += theta[i] * x[i]
        return sum

    def cost_f(theta, X, Y):
        m = len(Y)
        sum = 0
        for i in range(m):
            sum += (h(theta, X[i]) - Y[i]) ** 2

        return sum / (2 * m)

    def der_theta(theta, X, Y, alpha, theta_idx):
        m = len(Y)
        sum = 0
        for i in range(m):
            sum += (h(theta, X[i]) - Y[i]) * X[i][theta_idx]

        return (alpha / m) * sum

    ## GD algorithm
    theta_history = np.zeros([iterations + 1, features_count])
    cost_history = np.zeros(iterations + 1)  # .reshape(1, -1) if required

    cost = cost_f(theta, X, Y)

    cost_history[0] = cost
    theta_history[0] = theta

    for it_idx in range(iterations):
        tmptheta = np.copy(theta)

        for theta_idx in range(features_count):
            tmptheta[theta_idx] = theta[theta_idx] - der_theta(theta, X, Y, alpha, theta_idx)

        theta = np.copy(tmptheta)

        cost = cost_f(theta, X, Y)

        cost_history[it_idx + 1] = cost
        theta_history[it_idx + 1] = theta

    return theta, cost, theta_history, cost_history


de_normalized_df = original_df.drop('price', axis=1)
print(de_normalized_df.head())

X_denorm = de_normalized_df.copy()
X_denorm.insert(0, '', 1)
X_denorm = X_denorm.values.astype('float64')

print(X_denorm[0: 2])

# Mean normalization
normalized_df = (de_normalized_df-de_normalized_df.mean())/(de_normalized_df.max()-de_normalized_df.min())
print(normalized_df.head())

X_norm = normalized_df.copy()
X_norm.insert(0, '', 1)
X_norm = X_norm.values.astype('float64')

print(X_norm[0: 2])

# RuntimeWarning: overflow encountered in double_scalars
theta, cost, _, denorm_cost_history = gd_old(X_denorm, Y)

print(f'Cost: {cost}')
print(f'Theta: {theta}')

theta, cost, _, norm_cost_history = gd_old(X_norm, Y)

print(f'Cost: {cost}')
print(f'Theta: {theta}')


plt.figure(figsize=(10,6))

plt.plot(range(len(denorm_cost_history)), denorm_cost_history, label='Денормальизованные данные')
plt.plot(range(len(norm_cost_history)), norm_cost_history, label='Нормальизованные данные')

plt.grid(True)
plt.xlabel("Итерации")
plt.ylabel("Значение функции стоимости")
plt.legend()
plt.show()

# 8 Реализуйте функции потерь J(θ) и градиентного спуска для случая многомерной линейной регрессии
# с использованием векторизации.

def h(theta, x):
    return np.dot(x, theta.T)


def cost_f(theta, X, Y):
    m = len(Y)
    results = h(theta, X) - Y
    return (np.dot(results.T, results) / (2 * m)).item()


def gd_vec(X, Y_un_resh, iterations=200, alpha=0.01):
    Y = Y_un_resh.reshape(-1, 1)

    features_count = X.shape[1]
    m = Y.size

    theta = np.zeros([1, features_count])

    theta_history = np.zeros([iterations + 1, features_count])
    cost_history = np.zeros(iterations + 1)

    cost = cost_f(theta, X, Y)

    cost_history[0] = cost
    theta_history[0] = theta

    for it_idx in range(iterations):
        dt = np.dot((h(theta, X) - Y).T, X)
        theta = theta - (alpha / m) * dt

        cost = cost_f(theta, X, Y)

        cost_history[it_idx + 1] = cost
        theta_history[it_idx + 1] = theta

    return theta, cost, theta_history, cost_history


theta, cost, theta_history, cost_history = gd_old(X_norm, Y)

print(f'Cost: {cost}')
print(f'Theta: {theta}')

theta, cost, theta_history, cost_history = gd_vec(X_norm, Y)

print(f'Cost: {cost}')
print(f'Theta: {theta}')

# 9 Покажите, что векторизация дает прирост производительности.

start = time.time()
theta, _, _, _ = gd_old(X_norm, Y, 9000, 0.1)
end = time.time()
print("Time:")
print(end - start)
print(theta)

start = time.time()
theta, _, _, _ = gd_vec(X_norm, Y, 9000, 0.1)
end = time.time()
print("Time:")
print(end - start)
print(theta)

# 10 Попробуйте изменить параметр ɑ (коэффициент обучения).
# Как при этом изменяется график функции потерь в зависимости от числа итераций градиентного спуск?
# Результат изобразите в качестве графика.

loss = {}
iterations_count = 120

for alpha in [0.01, 0.05, 0.1, 0.5, 1, 2]:
    _, _, _, cost_history = gd_vec(X_norm, Y, iterations_count, alpha)
    loss[str(alpha)] = cost_history

plt.figure(figsize=(10,6))

iter_range = range(iterations_count + 1)

for alpha, cost_series in loss.items():
    plt.plot(iter_range, cost_series, label=r'$ \alpha $: %s'%(alpha))

plt.grid(True)
plt.xlabel("Итерации")
plt.ylabel("Значение функции стоимости")
plt.legend()
plt.show()

# 11 Постройте модель, используя аналитическое решение, которое может быть получено методом наименьших квадратов.
# Сравните результаты данной модели с моделью, полученной с помощью градиентного спуска.

def normal_equation(X, Y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)

start = time.time()
normal_equation(X_norm, Y)
end = time.time()
print("Time:")
print(end - start)
print(theta)

start = time.time()
theta, _, _, _ = gd_vec(X_norm, Y, 9000, 0.1)
end = time.time()
print("Time:")
print(end - start)
print(theta)