import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.io
import time

# 1. Загрузите данные ex2data1.txt из текстового файла.

df1 = pandas.read_csv('ex2data1.txt', header=None, names=['exam_1', 'exam_2', 'result'])
print(df1.head())

# 2. Постройте график, где по осям откладываются оценки по предметам, а точки обозначаются двумя разными маркерами
# в зависимости от того, поступил ли данный студент в университет или нет.

def create_normalizer(df):
    mean = df.mean().values
    rng = (df.max() - df.min()).values

    def norm_func(val):
        return (val - mean) / rng

    def denorm_func(val):
        return val * rng + mean

    return norm_func, denorm_func

tmp_df = df1.copy()
tmp_df.insert(0, '', 1)

orig_X = tmp_df.drop('result', axis=1).values
print(f'orig_X.shape = {orig_X.shape}')


norm_func, denorm_func = create_normalizer(df1.copy().drop('result', axis=1))

norm_df = norm_func(df1.copy().drop('result', axis=1))
norm_df.insert(0, '', 1)

norm_X = norm_df.values
print(f'norm_X.shape = {norm_X.shape}')

Y = tmp_df['result'].values.reshape(-1, 1)
print(f'Y.shape = {Y.shape}')

r_pass = Y == 1
r_fail = Y == 0

plt.figure(figsize=(10,6))
plt.plot(df1[r_pass]['exam_1'],df1[r_pass]['exam_2'], 'go', label='Поступил')
plt.plot(df1[r_fail]['exam_1'],df1[r_fail]['exam_2'], 'rx', label='Не поступил')
plt.xlabel('Экзамен 1')
plt.ylabel('Экзамен 2')
plt.legend(loc=3)
plt.grid(True)
plt.show()

# 3. Реализуйте функции потерь J(θ) и градиентного спуска для логистической регрессии с использованием векторизации.

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def h(theta, X):
    return sigmoid(np.dot(X, theta))


def J(theta, X, Y):
    m = len(X)

    h_res = h(theta, X)

    e1 = np.dot(-Y.T, np.log(h_res))
    e2 = np.dot((1 - Y).T, np.log(1 - h_res))

    return (1 / m) * (e1 - e2).item()


initial_theta = np.zeros((orig_X.shape[1], 1))
print(J(initial_theta, orig_X, Y))


def gd(X, Y, iterations=200, alpha=0.01):
    n = X.shape[1]
    m = len(Y)

    theta = np.zeros([n, 1])

    j_hist = []

    for i in range(iterations):
        h_res = h(theta, X)

        dt = np.dot(X.T, (h_res - Y))

        theta = theta - (alpha / m) * dt

        j_hist.append(J(theta, X, Y))

    return theta, np.asarray(j_hist)

gd_theta, j_hist = gd(norm_X, Y, 20000, 0.5)

print(f'\u03B8 = {gd_theta.reshape(-1)}')
print(f'J(\u03B8) = {j_hist[-1]}')

plt.figure(figsize=(10,6))
plt.plot(range(len(j_hist)), j_hist)
plt.grid(True)
plt.xlabel("Итерации")
plt.ylabel("J(\u03B8)")
plt.show()

# 4. Реализуйте другие методы (как минимум 2) оптимизации для реализованной функции стоимости
# (например, Метод Нелдера — Мида, Алгоритм Бройдена — Флетчера — Гольдфарба — Шанно, генетические методы и т.п.).
# Разрешается использовать библиотечные реализации методов оптимизации (например, из библиотеки scipy).

def nelder_mead(X, Y):
    result = optimize.minimize(J, x0=np.zeros([X.shape[1], 1]), args=(X, Y), method='Nelder-Mead')
    return result.x, result.fun

start = time.time()
nm_theta, nm_cost = nelder_mead(norm_X, Y)
end = time.time()
print("Time: ")
print(end - start)
print(f'\u03B8 = {nm_theta}')
print(f'J(\u03B8) = {nm_cost}')

def bfgs(X, Y):
    result = optimize.minimize(J, x0=np.zeros([X.shape[1], 1]), args=(X, Y), method='BFGS')
    return result.x, result.fun

start = time.time()
bfg_theta, bfg_cost = bfgs(norm_X, Y)
end = time.time()
print("Time: ")
print(end - start)
print(f'\u03B8 = {nm_theta}')
print(f'J(\u03B8) = {nm_cost}')


# def bfgs_cust(X, Y):
#     def jac_cust(theta, X, Y):
#         m = len(Y)
#         h_res = h(theta, X)
#         dt = np.dot((h_res - Y).T, X)
#         res = ((1 / m) * dt)
#         return res
#
#     init_theta = np.zeros([X.shape[1], 1]).flatten()
#
#     result = optimize.minimize(J, x0=init_theta, args=(X, Y.flatten()), method='BFGS', jac=jac_cust)
#     return result.x, result.fun
#
# start = time.time()
# bfg_c_theta, bfg_c_cost = bfgs_cust(norm_X, Y)
# end = time.time()
# print("Time: ")
# print(end - start)
# print(f'\u03B8 = {nm_theta}')
# print(f'J(\u03B8) = {nm_cost}')

# 5. Реализуйте функцию предсказания вероятности поступления студента в зависимости от значений оценок по экзаменам.

def predict(theta, x):
    new_x = np.insert(norm_func(x), 0, 1, axis=0).reshape(1, -1)
    return (h(theta, new_x) >= 0.5).astype(int)

print(predict(gd_theta, [70, 55]).item())

# 6. Постройте разделяющую прямую, полученную в результате обучения модели. Совместите прямую с графиком из пункта 2.

boundary_x = np.array([np.min(norm_X[:,1]), np.max(norm_X[:,1])])
boundary_y = (-1./gd_theta[2])*(gd_theta[0] + gd_theta[1]*boundary_x)


line =  np.hstack((boundary_x.reshape(-1, 1), boundary_y.reshape(-1, 1)))
line = denorm_func(line)


plt.figure(figsize=(10,6))
plt.plot(df1[r_pass]['exam_1'], df1[r_pass]['exam_2'], 'go', label='Поступил')
plt.plot(df1[r_fail]['exam_1'], df1[r_fail]['exam_2'], 'rx', label='Не поступил')
plt.plot(line[:,0], line[:,1],'b-',label='Граница')
plt.xlabel('Экзамен 1')
plt.ylabel('Экзамен 2')
plt.legend(loc=3)
plt.grid(True)
plt.show()

# 7. Загрузите данные ex2data2.txt из текстового файла.

ex2_df = pandas.read_csv('ex2data2.txt', header=None, names=['test_1', 'test_2', 'result'])
print(ex2_df.head())

tmp_ex2_df = ex2_df.copy()

X2 = tmp_ex2_df.drop('result', axis=1).values
print(f'X2.shape = {X2.shape}')

Y2 = tmp_ex2_df['result'].values.reshape(-1, 1)
print(f'Y2.shape = {Y2.shape}')

r2_pass = Y2 == 1
r2_fail = Y2 == 0

# 8. Постройте график, где по осям откладываются результаты тестов, а точки обозначаются двумя разными маркерами
# в зависимости от того, прошло ли изделие контроль или нет.

def draw_ex_2_data():
    plt.figure(figsize=(8,8))

    plt.plot(ex2_df[r2_pass]['test_1'], ex2_df[r2_pass]['test_2'], 'go', label='Прошел')
    plt.plot(ex2_df[r2_fail]['test_1'], ex2_df[r2_fail]['test_2'], 'rx', label='Не прошел')
    plt.xlabel('Тест 1')
    plt.ylabel('Тест 2')
    plt.legend()
    plt.grid(True)
    plt.show()
draw_ex_2_data()

# 9. Постройте все возможные комбинации признаков x1 (результат первого теста) и x2 (результат второго теста),
# в которых степень полинома не превышает 6, т.е. 1, x1, x2, x12, x1x2, x22, …, x1x25, x26 (всего 28 комбинаций).

def gen_polynom_matrix(X, degrees):
    # первый столбик с единицами
    m = len(X)
    result = np.ones([m, 1])

    for i in range(1, degrees + 1):
        for j in range(0, i + 1):
            x1 = X[:, 0] ** (i - j)
            x2 = X[:, 1] ** (j)
            new_column = (x1 * x2).reshape(m, 1)
            result = np.hstack((result, new_column))

    return result

X2_new = gen_polynom_matrix(X2, 6)
print(X2_new.shape)

# 10. Реализуйте L2-регуляризацию для логистической регрессии и обучите ее на расширенном наборе признаков методом
# градиентного спуска.


def J_reg(theta, X, Y, l=0):
    m = len(X)

    h_res = h(theta, X)

    e1 = np.dot(-Y.T, np.log(h_res))
    e2 = np.dot((1 - Y).T, np.log(1 - h_res))

    reg = l / 2 * np.sum(np.dot(theta[1:].T, theta[1:]))

    return (1 / m) * ((e1 - e2) + reg).item()


print(J_reg(np.asarray([0, 0.5, 3]), norm_X, Y, 0.1))


def gd_step(theta, X, Y, lam=0):
    m = len(Y)
    h_res = h(theta, X)

    res = (1. / m) * np.dot(X.T, (h_res - Y))

    res[1:] = res[1:] + (lam / m) * theta[1:]

    return res


print(gd_step(np.zeros([X2.shape[1], 1]), X2, Y2))


def gd_reg(X, Y, iterations=200, alpha=0.01, l=0):
    n = X.shape[1]
    m = len(Y)

    theta = np.zeros([n, 1])

    j_hist = []

    for i in range(iterations):
        theta = theta - alpha * gd_step(theta, X, Y, l)
        j_hist.append(J_reg(theta, X, Y, l))

    return theta, np.asarray(j_hist)

gd_reg_theta, j_reg_hist = gd_reg(X2, Y2, 10000)

print(f'\u03B8 = {gd_reg_theta.reshape(-1)}')
print(f'J(\u03B8) = {j_reg_hist[-1]}')

gd_reg_theta_new, j_reg_hist_new = gd_reg(X2_new, Y2, 100)

print(f'\u03B8 = {gd_reg_theta_new.reshape(-1)}')
print(f'J(\u03B8) = {j_reg_hist_new[-1]}')

# 11. Реализуйте другие методы оптимизации.

def bfgs_cust_reg(X, Y, l=0):
    init_theta = np.zeros([X.shape[1], 1]).flatten()

    result = optimize.minimize(J_reg, x0=init_theta, args=(X, Y.flatten(), l), method='BFGS', jac=gd_step)
    return result.x, result.fun


print(bfgs_cust_reg(X2_new, Y2))


# 12. Реализуйте функцию предсказания вероятности прохождения контроля изделием в зависимости от результатов тестов.

def predict_2(theta, x):
    return (h(theta, x) >= 0.5).astype(int)

print(f'Should equal to 1 - {predict_2(gd_reg_theta, [-0.25, 0.5]).item()}')
print(f'Should equal to 0 - {predict_2(gd_reg_theta, [1, 1]).item()}')


# 13. Постройте разделяющую кривую, полученную в результате обучения модели. Совместите прямую с графиком из пункта 7.

def plotBoundary(X, Y, l=0.):
    theta, j_h = bfgs_cust_reg(X, Y, l)
    # theta, j_h = gd_reg(X, Y,iterations = 12000, alpha = 0.01, l=l)

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            fearure_row = gen_polynom_matrix(np.array([[u[i], v[j]]]), 6)
            z[i][j] = h(theta, fearure_row)

    z = z.T

    draw_ex_2_data()
    c = plt.contour(u, v, z, 0, colors='blue')
    c.collections[0].set_label(f'\u03bb = {l}')
    print(c.collections)
    plt.legend()
    plt.title("Разделяющая кривая")
    plt.show()

plotBoundary(X2_new, Y2, 0.001)

# 14. Попробуйте различные значения параметра регуляризации λ. Как выбор данного значения влияет на вид
# разделяющей кривой? Ответ дайте в виде графиков.

plotBoundary(X2_new, Y2, 0.01)
plotBoundary(X2_new, Y2, 0.1)
plotBoundary(X2_new, Y2, 20)

# 15. Загрузите данные ex2data3.mat из файла.

img_data = scipy.io.loadmat('ex2data3.mat')
ex_3_X, ex_3_Y = img_data['X'], img_data['y']
print(ex_3_X.shape)

# 16. Визуализируйте несколько случайных изображений из набора данных. Визуализация должна содержать каждую цифру как
# минимум один раз.

vals, indexes = np.unique(ex_3_Y, return_index=True)
print(vals)

def getImgFromRow(row):
    return row.reshape(20, 20).T


fig, axs = plt.subplots(1, 10)

for i in range(len(indexes)):
    index = indexes[i]
    val = vals[i]
    axs[i].imshow(getImgFromRow(ex_3_X[index]), cmap='gray')
    axs[i].axis("off")

plt.show()

# 17. Реализуйте бинарный классификатор с помощью логистической регрессии с использованием векторизации
# (функции потерь и градиентного спуска).
# 17. Добавьте L2-регуляризацию к модели.

ex_3_m = len(ex_3_Y)
ex_3_n = ex_3_X.shape[1]

print(f'm = {ex_3_m}, n = {ex_3_n}')

ex_3_init_theta = np.zeros([ex_3_n, 1])
print(ex_3_init_theta.shape)

# 19. Реализуйте многоклассовую классификацию по методу “один против всех”.


ex_3_X_ext = np.hstack((np.ones((ex_3_m, 1)), ex_3_X))

def fmin_cg_alg(X, Y, lam=0):
    init_theta = np.zeros([X.shape[1], 1]).flatten()
    result = optimize.fmin_cg(J_reg, fprime=gd_step, x0=init_theta, args=(X, Y.flatten(), lam), maxiter=50, disp=False, full_output=True)
    return result[0], result[1]


def train_classifier(X, Y, lam=0):
    m = X.shape[0]
    n = X.shape[1]

    classes_count = 10

    thetas = np.zeros([classes_count, n])

    for klass in range(classes_count):
        class_index = klass if klass else 10  # 10 - это 0
        print(f'{klass} => {class_index}')
        replaced_Y = (Y == class_index).astype(int)
        # theta, cost = fmin_cg_alg(X, replaced_Y, lam)  # lam - 0.001 -> 95.6%
        theta, cost = bfgs_cust_reg(X, replaced_Y, lam)  # lam - 0.001 -> 97.2%
        thetas[klass] = theta

    return thetas


thetas = train_classifier(ex_3_X_ext, ex_3_Y, 0.001)
print(thetas.shape)

# 20. Реализуйте функцию предсказания класса по изображению с использованием обученных классификаторов.

def predClass(thetas, x):
    return np.argmax(h(thetas.T, x))

i = 3000
x = ex_3_X_ext[i]
y = ex_3_Y[i].item()

print(f'Class = {y}, predicted = {predClass(thetas, x)}')

# 21. Процент правильных классификаций на обучающей выборке должен составлять около 95%.

def calc_accuracy(thetas, X, Y):
    m = X.shape[0]
    correct = 0

    for i in range(m):
        pred = predClass(thetas, X[i])
        pred = pred if pred else 10

        if pred == Y[i]:
            correct += 1

    return correct / m

print("Accuracy: %0.1f%%" % (100 * calc_accuracy(thetas, ex_3_X_ext, ex_3_Y)))