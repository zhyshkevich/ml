import scipy.io
import numpy as np
from itertools import chain
from scipy import optimize
import matplotlib.pyplot as plt


# 1. Загрузите данные ex4data1.mat из файла.

img_data = scipy.io.loadmat('ex4data1.mat')

X, Y = img_data['X'], img_data['y']

print(f'X.shape = {X.shape}')
print(f'Y.shape = {Y.shape}')

# 2. Загрузите веса нейронной сети из файла ex4weights.mat, который содержит две матрицы Θ(1) (25, 401) и Θ(2) (10, 26).
# Какова структура полученной нейронной сети?

weights_data = scipy.io.loadmat('ex4weights.mat')
theta1 = weights_data['Theta1']
theta2 = weights_data['Theta2']

print(f'theta1.shape = {theta1.shape}')
print(f'theta2.shape = {theta2.shape}')

nn_params = {
    'layer_1_input': theta1.shape[1],
    'layer_1_output': theta1.shape[0],
    'layer_2_input': theta2.shape[1],
    'layer_2_output': theta2.shape[0],
}

print(nn_params)

def unroll_thetas(thetas):
    return np.concatenate([thetas[0].flatten(), thetas[1].flatten()])


def rehape_thetas(unrolled_thetas):
    theta1_start = 0
    theta1_end = theta1_start + (nn_params['layer_1_input'] * nn_params['layer_1_output'])
    theta2_start = theta1_end
    theta2_end = theta2_start + (nn_params['layer_2_input'] * nn_params['layer_2_output'])

    theta1 = unrolled_thetas[theta1_start:theta1_end].reshape((nn_params['layer_1_output'], nn_params['layer_1_input']))
    theta2 = unrolled_thetas[theta2_start:theta2_end].reshape((nn_params['layer_2_output'], nn_params['layer_2_input']))

    return np.array([theta1, theta2])


thetas = np.array([theta1, theta2])


# 3. Реализуйте функцию прямого распространения с сигмоидом в качестве функции активации.

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def add_bias_vec(a):
    return np.insert(a,0,1,axis=1)

def rm_bias(input):
    return input[:, 1:]


def forward_prop_vec_all(thetas, X):
    a1 = add_bias_vec(X)
    z2 = np.dot(a1, thetas[0].T)
    a2 = sigmoid(z2)

    a2 = add_bias_vec(a2)
    z3 = np.dot(a2, thetas[1].T)
    a3 = sigmoid(z3)

    return {'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'a3': a3}


def forward_prop_vec(thetas, X):
    return forward_prop_vec_all(thetas, X)['a3']

print(forward_prop_vec(thetas, X).shape)

# 4. Вычислите процент правильных классификаций на обучающей выборке. Сравните полученный результат с
# логистической регрессией.

def predict(thetas, x):
    if x.ndim == 1:
        x = x.reshape(1, -1)

    fp_res = forward_prop_vec(thetas, x)

    return np.argmax(fp_res[0]) + 1


def calc_accuracy(thetas, X, Y):
    m = X.shape[0]
    correct = 0

    for i in range(m):
        if predict(thetas, X[i]) == Y[i]:
            correct += 1

    return correct / m

print("Accuracy: %0.1f%%"%(100*calc_accuracy(thetas, X, Y)))


# Для лог-регрессии было 97.2%

# 5. Перекодируйте исходные метки классов по схеме one-hot.

def one_hot(labels):
    m = len(labels)
    uniq_labels = np.unique(labels)

    return (labels == uniq_labels).astype(int)

Y_oh = one_hot(Y)
print(Y_oh[0])

# 6. Реализуйте функцию стоимости для данной нейронной сети

def J(thetas, X, Y_one_hot, lmb=0.):
    m = len(X)
    h = forward_prop_vec(thetas, X)

    e1 = np.multiply(Y_one_hot, np.log(h))
    e2 = np.multiply((1 - Y_one_hot), np.log(1 - h))

    regularization = 0
    cost = (-1 / m) * np.sum(e1 + e2)

    # Не помню, нужно ли в регуляризации учитывать баес? В Лог регрессии - не нужно было
    if lmb != 0:
        reg_sum = np.sum(np.power(rm_bias(thetas[0]), 2)) + np.sum(np.power(rm_bias(thetas[1]), 2))
        regularization = (lmb / (2 * m)) * reg_sum

    return cost + regularization


def J_unroll(unroll_thetas, X, Y_one_hot, lmb=0.):
    return J(rehape_thetas(unroll_thetas), X, Y_one_hot, lmb)

print(J(thetas, X, Y_oh))
print(J_unroll(unroll_thetas(thetas), X, Y_oh))


# 7. Добавьте L2-регуляризацию в функцию стоимости.

def der_sigmoid(a):
    return np.multiply(a, 1 - a)

print(der_sigmoid(np.array([1, 2, 3])))

# 9. Инициализируйте веса небольшими случайными числами.

def gen_thetas(eps=0.1):
    t1 = np.random.rand(nn_params['layer_1_output'], nn_params['layer_1_input']) * (2 * eps) - eps
    t2 = np.random.rand(nn_params['layer_2_output'], nn_params['layer_2_input']) * (2 * eps) - eps

    return np.array([t1, t2])

new_thetas = gen_thetas()

print("Rand thetas accuracy: %0.1f%%"%(100*calc_accuracy(new_thetas, X, Y)))

# Rand thetas accuracy: 10.0%

# 10. Реализуйте алгоритм обратного распространения ошибки для данной конфигурации сети.

def back_prop_vec(unrolled_thetas, X, Y_one_hot, lmb=0.):
    # print('call - back_prop_vec')
    m = len(X)
    thetas = rehape_thetas(unrolled_thetas)

    delta_1 = np.zeros(thetas[0].shape)
    delta_2 = np.zeros(thetas[1].shape)

    for i in range(m):
        fp_data = forward_prop_vec_all(thetas, X[i].reshape(1, -1))

        d3 = fp_data['a3'] - Y_one_hot[i]  # слой выхода

        d2 = rm_bias(np.multiply(np.dot(d3, thetas[1]), der_sigmoid(fp_data['a2'])))

        delta_1 += np.dot(d2.T, fp_data['a1'])
        delta_2 += np.dot(d3.T, fp_data['a2'])

    delta_1 /= m
    delta_2 /= m

    if lmb != 0:
        lmb_mult = (lmb / m)
        delta_1[:, 1:] += lmb_mult * rm_bias(thetas[0])
        delta_2[:, 1:] += lmb_mult * rm_bias(thetas[1])

    return unroll_thetas(np.array([delta_1, delta_2]))

back_prop_thetas = back_prop_vec(unroll_thetas(thetas), X, Y_oh)
print(back_prop_thetas)


# 11. Для того, чтобы удостоверится в правильности вычисленных значений градиентов используйте метод проверки градиента
# с параметром ε = 10-4.

def gd_check(experiments_count, unrolled_thetas, back_prop_thetas, X, Y_one_hot, lmb=0.):
    eps = 0.0001
    theta_count = len(unrolled_thetas)

    for i in range(experiments_count):
        idx = int(np.random.rand() * theta_count)

        experiment_thetas = np.copy(unrolled_thetas)
        orig_val = experiment_thetas[idx]

        experiment_thetas[idx] = orig_val + eps
        cost_plus = J_unroll(experiment_thetas, X, Y_one_hot, lmb)

        experiment_thetas[idx] = orig_val - eps
        cost_minus = J_unroll(experiment_thetas, X, Y_one_hot, lmb)

        calc_g = (cost_plus - cost_minus) / (2 * eps)

        print(f'Idx: {idx} check gradient: {calc_g:f}, BP gradient: {back_prop_thetas[idx]:f}')

gd_check(5, unroll_thetas(thetas), back_prop_thetas, X, Y_oh, 0)


# 12. Добавьте L2-регуляризацию в процесс вычисления градиентов.

back_prop_thetas_reg = back_prop_vec(unroll_thetas(thetas), X, Y_oh, 0.5)
print(back_prop_thetas_reg)

# 13. Проверьте полученные значения градиента.

gd_check(5, unroll_thetas(thetas), back_prop_thetas_reg, X, Y_oh, 0.5)

# 14. Обучите нейронную сеть с использованием градиентного спуска или других более эффективных методов оптимизации.

def fit(X, Y, lmb=0., maxiter=30):
    rand_thetas = gen_thetas()
    unrolled_thetas = unroll_thetas(rand_thetas)

    Y_one_hot = one_hot(Y)

    # back_prop_vec(unroll_thetas(thetas), X, Y_oh, 0.5)
    # result = optimize.minimize(J_unroll, jac=back_prop_vec, x0=unrolled_thetas, args=(X, Y_one_hot, lmb), method='BFGS', options={ 'maxiter': maxiter , 'disp': True})
    # out_thetas = rehape_thetas(result.x)
    result = optimize.fmin_cg(maxiter=maxiter, f=J_unroll, x0=unrolled_thetas, fprime=back_prop_vec,
                              args=(X, Y_one_hot, lmb))
    out_thetas = rehape_thetas(result)

    return out_thetas

fitted_thetas = fit(X, Y, maxiter = 75)

# 15. Вычислите процент правильных классификаций на обучающей выборке.

print("NN accuracy: %0.1f%%"%(100*calc_accuracy(fitted_thetas, X, Y)))

# 16. Визуализируйте скрытый слой обученной сети.


def get_img_from_row(row):
    return row.reshape(20, 20).T


def show_images(thetas):
    images = thetas[0][:, 1:]

    fig, axs = plt.subplots(5, 5, figsize=(8, 8))
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    axs = axs.flatten()

    for i in range(len(axs)):
        ax = axs[i]
        ax.imshow(get_img_from_row(images[i]), cmap='gray')
        ax.axis('off')

    plt.show()

show_images(fitted_thetas)

# 17. Подберите параметр регуляризации. Как меняются изображения на скрытом слое в зависимости от данного параметра?

lambda_values = [0, 0.001, 0.01, 1., 10., 30.]

data_dict = {}
for i in range(len(lambda_values)):
    lmbd = lambda_values[i]

    t = fit(X, Y, lmb=lmbd, maxiter=45)
    ac = calc_accuracy(t, X, Y)

    data_dict[str(lmbd)] = {
        'lambda': lmbd,
        'thetas': t,
        'accuracy': ac
    }

lambdas = []
accuracies = []

for key, value in data_dict.items():
    print(f"lambda: {key}, accuracy: {value['accuracy']}")
    lambdas.append(value['lambda'])
    accuracies.append(value['accuracy'])

    show_images(value['thetas'])

xVals = np.arange(len(lambdas))

plt.figure(figsize=(10,6))
plt.plot(xVals, accuracies, 'o-')
plt.grid(True)
plt.xlabel("\u03BB", fontsize=25)
plt.ylabel("Accuracy", fontsize=25)
plt.xticks(xVals, lambdas)
plt.show()