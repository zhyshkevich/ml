import scipy.io
from scipy import optimize
import numpy as np

# 1. Загрузите данные ex9_movies.mat из файла.

movie_data = scipy.io.loadmat('ex9_movies.mat')
Y = movie_data['Y']
R = movie_data['R']

Nm, Nu = Y.shape

print(f'Y.shape = {Y.shape}')
print(f'R.shape = {R.shape}')

print(f'Users: {Nu}, Movies: {Nm}')

# 2. Выберите число признаков фильмов (n) для реализации алгоритма коллаборативной фильтрации.

x_feat_count = 4

# 3. Реализуйте функцию стоимости для алгоритма.

def h(theta, X):
    return np.dot(X, theta.T)


def cost(theta, X, y, r):
    y_pred = h(theta, X)
    y_pred = y_pred * r

    return np.sum(np.power((y_pred - y), 2))


def J_combined(theta, X, y, r, lmb=0.):
    reg = 0
    error = cost(theta, X, y, r)

    if lmb != 0:
        reg += lmb * np.sum(np.square(theta))
        reg += lmb * np.sum(np.square(X))

    return (error + reg) / 2

# 4. Реализуйте функцию вычисления градиентов.

def unroll(theta, X):
    return np.concatenate([theta.flatten(), X.flatten()])


def roll_up(data, movies_c, users_c, feat_c):
    theta_end = users_c * feat_c
    theta = data[:theta_end].reshape(users_c, feat_c)
    X = data[theta_end:].reshape(movies_c, feat_c)

    return theta, X


def J_gd(data, Y, R, Nm, Nu, Nf, lmb=0.):
    theta, X = roll_up(data, Nm, Nu, Nf)

    return J_combined(theta, X, Y, R, lmb)


def gd_step(data, Y, R, Nm, Nu, Nf, lmb=0.):
    theta, X = roll_up(data, Nm, Nu, Nf)

    error = (h(theta, X) * R) - Y

    X_gd = np.dot(error, theta)
    theta_gd = np.dot(error.T, X)

    if lmb != 0:
        X_gd += lmb * X
        theta_gd += lmb * theta

    return unroll(theta_gd, X_gd)

# 7. Обучите модель с помощью градиентного спуска или других методов оптимизации.

def build_model(init_theta, init_X, Y, R, Nm, Nu, Nf, lmb=0.):
    data = optimize.fmin_cg(
        J_gd,
        x0=unroll(init_theta, init_X),
        fprime=gd_step,
        args=(Y, R, Nm, Nu, Nf, lmb),
        maxiter=400,
        disp=True
    )

    return roll_up(data, Nm, Nu, Nf)

rand_theta = np.random.rand(Nu, x_feat_count)
rand_X = np.random.rand(Nm, x_feat_count)
calc_theta, calc_X = build_model(rand_theta, rand_X, Y, R, Nm, Nu, x_feat_count)

# 8. Добавьте несколько оценок фильмов от себя. Файл movie_ids.txt содержит индексы каждого из фильмов.

def load_vocab(filename):
    dict = {}
    with open(filename, encoding = "ISO-8859-1") as f:
        for line in f:
            (idx, name) = line.strip().split(' ', 1)
            dict[int(idx) -1] = name
    return dict

movies = load_vocab('movie_ids.txt')

def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / y_true.size


def recommend_movies(predictions, R, user_id, top=10):
    predictions = predictions[:, user_id]
    not_watched_R = np.where(R[:, user_id] < 1)[0]
    predictions = predictions[not_watched_R]
    recommended_idx = np.argsort(predictions)
    recommended_idx = recommended_idx[::-1]  # asc -> desc

    print(f'Top {top}:')
    for i in range(top):
        idx = recommended_idx[i]
        print(f'{predictions[idx]:.2f} - {movies[idx]}')

my_rank = np.zeros([Y.shape[0],1]).astype(int)

#1 Toy Story (1995)
my_rank[0] = 3
#2 GoldenEye (1995)
my_rank[1] = 5
#50 Star Wars (1977)
my_rank[49] = 3
#71 Lion King, The (1994)
my_rank[70] = 2
#72 Mask, The (1994)
my_rank[71] = 3
#96 Terminator 2: Judgment Day (1991)
my_rank[95] = 5
#204 Back to the Future (1985)
my_rank[203] = 3
#210 Indiana Jones and the Last Crusade (1989)
my_rank[209] = 5
#222 Star Trek: First Contact (1996)
my_rank[221] = 3
#227 Star Trek VI: The Undiscovered Country (1991)
my_rank[226] = 3
#228 Star Trek: The Wrath of Khan (1982)
my_rank[227] = 3
#229 Star Trek III: The Search for Spock (1984)
my_rank[228] = 3
#230 Star Trek IV: The Voyage Home (1986)
my_rank[229] = 3
#226 Die Hard 2 (1990)
my_rank[225] = 4
#405 Mission: Impossible (1996)
my_rank[404] = 5
#550 Die Hard: With a Vengeance (1995)
my_rank[549] = 5
#407 Spy Hard (1996)
my_rank[406] = 3
#384 Naked Gun 33 1/3: The Final Insult (1994)
my_rank[383] = 3
#455 Jackie Chan's First Strike (1996)
my_rank[454] = 5
#250 Fifth Element, The (1997)
my_rank[249] = 5

##

my_indexes = np.where(my_rank > 0)[0]

for idx in my_indexes:
    print(f"{movies[idx]}\t{my_rank[idx].item()}")

my_rated = (my_rank > 0).astype(int)

my_Y = np.c_[Y, my_rank]
print(my_Y.shape)

movies_mean = my_Y.mean(axis=1).reshape(-1, 1)
print(movies_mean.shape)

my_R = np.c_[R, my_rated]
print(my_R.shape)

my_Nm, my_Nu = my_Y.shape
my_feat_count = 100

init_theta = np.random.rand(my_Nu, my_feat_count)
init_X = np.random.rand(my_Nm, my_feat_count)

my_calc_theta, my_calc_X = build_model(init_theta, init_X, my_Y, my_R, my_Nm, my_Nu, my_feat_count, 0.2)
y_pred_gd = h(my_calc_theta, my_calc_X)

print(f'mse = {mse(my_Y, y_pred_gd)}')

recommend_movies(y_pred_gd, my_R, 943, 10)

# 10. Также обучите модель с помощью сингулярного разложения матриц. Отличаются ли полученные результаты?

def predict_using_svd(Y, feat_count):
    U, Sigma, Vt = np.linalg.svd(Y)

    U_f = U[:, : feat_count]
    S_f = np.diag(Sigma[: feat_count])
    Vt_f = Vt[: feat_count]

    predictions = np.dot(np.dot(U_f, S_f), Vt_f)

    return predictions

y_pred_svd = predict_using_svd(my_Y, 100)
print(f'mse = {mse(my_Y, y_pred_svd)}')

recommend_movies(y_pred_svd, my_R, 943, 10)






