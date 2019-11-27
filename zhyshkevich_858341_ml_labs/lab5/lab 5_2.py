import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn import svm
import nltk
from IPython.display import clear_output, display
import os
import email

# 13. Загрузите данные spamTrain.mat из файла.

data1 = scipy.io.loadmat('spamTrain.mat')
X1 = data1['X']
y1 = data1['y']

print(X1.shape)


# 14. Обучите классификатор SVM.

model1 = svm.SVC(C=0.5, kernel='linear')
model1.fit( X1, y1.flatten() )
print('spamTrain accuracy: ', (model1.score(X1, y1.flatten())) * 100)

# 15. Загрузите данные spamTest.mat из файла.

data2 = scipy.io.loadmat('spamTest.mat')
X1_test = data2['Xtest']
y1_test = data2['ytest']

print('spamTest accuracy: ', (model1.score(X1_test, y1_test.flatten())) * 100)

# 16. Подберите параметры C и σ2

def sigma_to_gamma_ang(sigma):
    return np.power(sigma, -2.) / 2

C_vals = [0.01, 0.04, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10, 20, 40, 80, 100]
sigma_vals = [0.01, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.4, 0.8, 1, 2, 5, 10, 20, 40, 80, 100]

# Для ядра гаусса

best_score = 0
params = {
    'C': 0,
    'sigma': 0
}

total_iter = len(C_vals) * len(sigma_vals)

i = 0
for C in C_vals:
    for sigma in sigma_vals:
        clear_output(wait=True)
        i += 1
        print(f'Current progress: {i}/{total_iter}')
        model = svm.SVC(C=C, kernel='rbf', gamma=sigma_to_gamma_ang(sigma))
        model.fit(X1, y1.flatten())
        score = model.score(X1_test, y1_test)
        if score > best_score:
            best_score = score
            params['C'] = C
            params['sigma'] = sigma

print(f"Лучшая комбинация набрала {best_score}: С = {params['C']}, sigma = {params['sigma']}")

# Для линейного ядра

best_score_linear = 0
params_linear_С = 0

total_iter_linear = len(C_vals)

i = 0
for C in C_vals:
    clear_output(wait=True)
    i += 1
    print(f'Current progress: {i}/{total_iter_linear}')
    model = svm.SVC(C=C, kernel='linear')
    model.fit(X1, y1.flatten())
    score = model.score(X1_test, y1_test)
    if score > best_score_linear:
        best_score_linear = score
        params_linear_С = C

print(f"Лучшая комбинация набрала {best_score_linear}: С = {params_linear_С}")

# 17. Реализуйте функцию предобработки текста письма, включающую в себя

def prepare_text(text):
    import re

    text = text.lower()  # перевод в нижний регистр;
    text = re.sub('<a .*href="([^"]*)".*?<\/a>', r' \1 ', text)  # достать ссылки из тегов
    text = re.sub('<[^<>]+>', ' ', text)  # удаление HTML тэгов
    text = re.sub('(http|https)://[^\s]*', ' httpaddr ', text)  # замена URL на одно слово (например, “httpaddr”)
    text = re.sub('[^\s]+@[^\s]+', ' emailaddr ', text)  # замена email-адресов на одно слово (например, “emailaddr”)
    text = re.sub('[0-9]+', ' number ', text)  # замена чисел на одно слово (например, “number”)
    text = re.sub('[$]+', ' dollar ', text)  # замена знаков доллара ($) на слово “dollar”
    # остальные символы должны быть удалены и заменены на пробелы, т.е. в результате получится текст, состоящий из слов, разделенных пробелами.
    text = re.sub('[_]+', ' ', text)
    text = re.sub('[^\w]', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    # замена форм слов на исходное слово (например, слова “discount”, “discounts”, “discounted”, “discounting” должны быть заменены на слово “discount”). Такой подход называется stemming;
    stemmer = nltk.stem.porter.PorterStemmer()
    words = text.split()
    stemmed_words = []
    for word in words:
        stemmed = stemmer.stem(word)
        if not len(stemmed): continue
        stemmed_words.append(stemmed)

    return ' '.join(stemmed_words)


prepare_text(
    "discounted №4843,,(03240.%\"№:304-235!\"№%:,.;()_+=_-OthEr things <p>p-tag</p><a href='https://somelink.com'>click here</a> THIS is some@tut.by 2019 and 7 he lost 15$ beautiful")

# 18. Загрузите коды слов из словаря vocab.txt.

def load_vocab(filename):
    dict = {}
    with open(filename) as f:
        for line in f:
            (val, key) = line.split()
            dict[key] = int(val)
    return dict

vocab = load_vocab('vocab.txt')

# 19. Реализуйте функцию замены слов в тексте письма после предобработки на их соответствующие коды.

def text_to_vec(text, vocab):
    n = len(vocab.keys())
    vec = np.zeros(n)

    words = prepare_text(text).split()
    for word in words:
        if word not in vocab: continue
        idx = vocab[word]
        # if idx >= n: continue # hack
        vec[idx - 1] = 1

    return vec.reshape(1, -1)

t_v = text_to_vec('here is my own email, beauti beautiful day of week', vocab)
print(np.unique(t_v, return_counts=True))
t_v.shape


# 20. Реализуйте функцию преобразования текста письма в вектор признаков (в таком же формате как в файлах
# spamTrain.mat и spamTest.mat).

def file_to_vec(filename, vocab):
    text = open(filename, 'r').read()

    return text_to_vec(text, vocab)

# 21. Проверьте работу классификатора на письмах из файлов emailSample1.txt, emailSample2.txt, spamSample1.txt
# и spamSample2.txt.

gauss_best_C = 5
gauss_best_sigma = 10

best_gaussian_model = svm.SVC(C=gauss_best_C, kernel='rbf', gamma=sigma_to_gamma_ang(gauss_best_sigma))
best_gaussian_model.fit(X1, y1.flatten())

best_C = 0.04

best_linear_model = svm.SVC(C=best_C, kernel='linear')
best_linear_model.fit(X1, y1.flatten())

emails_data = [
    ['emailSample1.txt', 0],
    ['emailSample2.txt', 0],
    ['spamSample1.txt', 1],
    ['spamSample2.txt', 1],
]


def predict_files(model, data):
    total = len(data)
    correct = 0
    for i in range(len(data)):
        item = data[i]
        email_vec = file_to_vec(item[0], vocab)
        y_pred = model.predict(email_vec)
        if (y_pred == item[1]): correct += 1
        print(f"File: {item[0]} - {item[1]}. Predicted: {y_pred}")

    accuracy = round(correct / total, 4)
    print('---------------------------')
    print(f'Accurecy: {accuracy * 100}%')

print('Predict with Linear kernel')
predict_files(best_linear_model, emails_data)

print('Predict with Gaussian kernel')
predict_files(best_gaussian_model, emails_data)

# 22. Также можете проверить его работу на собственных примерах.

def get_kaggle_emails(n=5):
    import pandas as pd
    kaggle_emails_df = pd.read_csv('emails.csv')

    spam_df = kaggle_emails_df[kaggle_emails_df['spam'] == 1]
    ham_df = kaggle_emails_df[kaggle_emails_df['spam'] == 0]

    output = []
    spam = spam_df['text'].values[:n]
    ham = ham_df['text'].values[:n]

    for i in range(len(ham)):
        output.append([ham[i], 0])

    for i in range(len(spam)):
        output.append([spam[i], 1])

    return output


def predict_text(model, data):
    total = len(data)
    correct = 0
    for i in range(len(data)):
        item = data[i]
        email_vec = text_to_vec(item[0], vocab)
        y_pred = model.predict(email_vec)
        if (y_pred == item[1]): correct += 1
        # print(f"Text: #{i} - {item[1]}. Predicted: {y_pred}")

    accuracy = round(correct / total, 4)
    print('---------------------------')
    print(f'Accurecy: {accuracy * 100}%')


kaggle_emails = get_kaggle_emails(1532)

kaggle_spam = 0
kaggle_ham = 0

for i in range(len(kaggle_emails)):
    y = kaggle_emails[i][1]
    if y == 1:
        kaggle_spam += 1
    if y == 0:
        kaggle_ham += 1

print(f'dataset: {len(kaggle_emails)},spam: {kaggle_spam}, ham: {kaggle_ham}')

print('Predict with Linear kernel')
predict_text(best_linear_model, kaggle_emails)

print('Predict with Gaussian kernel')
predict_text(best_gaussian_model, kaggle_emails)

# 23. Создайте свой набор данных из оригинального корпуса текстов - http://spamassassin.apache.org/old/publiccorpus/.


spam_path = 'spamassassin_publiccorpus/spam'
ham_path = 'spamassassin_publiccorpus/ham'


def folder_with_emails_to_list(folder):
    texts = []
    for filename in os.listdir(folder):
        content = open(os.path.join(folder, filename), 'r', encoding="ISO-8859-1").read()
        mail = email.message_from_string(content)

        while mail.is_multipart():
            mail = mail.get_payload(0)

        content = mail.get_payload()

        if not len(content): continue

        # payload = mail.get_payload()
        texts.append(prepare_text(content))

    return texts

spam_messages = folder_with_emails_to_list(spam_path)
print(f'Spam messages count: {len(spam_messages)}')

ham_messages = folder_with_emails_to_list(ham_path)
print(f'Ham messages count: {len(ham_messages)}')

all_messages = spam_messages + ham_messages

words_count_dict = {}

#trash_words = ['aa', 'ac', 'ad', 'ae', 'af', 'ag', 'al', 'am', 'an', 'aw', 'ba', 'bb', 'bc', 'bd', 'be', 'bf', 'c', 'ca', 'cb', 'cc', 'cd', 'ce', 'cf', 'cj', 'cm', 'co']

for message in all_messages:
    words = message.split()
    for word in words:
        #if word in trash_words: continue
        #if (len(word) == 2): continue
        if word not in words_count_dict:
            words_count_dict[word] = 0
        words_count_dict[word] += 1

len(words_count_dict.keys())

most_freq = sorted(words_count_dict, key=words_count_dict.get, reverse=True)[:1899]


for i in range(20):
    word = most_freq[i]
    print(f'{word} => {words_count_dict[word]}')


# 24. Постройте собственный словарь.

most_freq.sort()

with open('my_vocab.txt', 'w+') as file:
    for i in range(len(most_freq)):
        word = most_freq[i]
        index = i+1
        file.write(f"{index}\t{word}\n")


# 25. Как изменилось качество классификации? Почему?

my_vocab = load_vocab('my_vocab.txt')
spam_messages_count = len(spam_messages)
ham_messages_count = len(ham_messages)
m = spam_messages_count + ham_messages_count
n = len(my_vocab.keys())
print(f'm = {m}, n = {n}')

my_X = np.zeros([m, n])
my_y = np.zeros(m)

print(f'my_X.shape = {my_X.shape}')
print(f'my_X.shape = {my_y.shape}')

for i in range(spam_messages_count):
    msg = spam_messages[i]
    my_X[i] = text_to_vec(msg, my_vocab)
    my_y[i] = 1

for i in range(ham_messages_count):
    msg = ham_messages[i]
    idx = spam_messages_count + i
    my_X[idx] = text_to_vec(msg, my_vocab)
    my_y[idx] = 0

np.unique(my_y, return_counts=True)

from sklearn.model_selection import train_test_split

my_X_train, my_X_test, my_y_train, my_y_test = train_test_split(my_X, my_y, test_size=0.3, random_state=10)

np.unique(my_y_test, return_counts=True)

my_best_score_linear = 0
my_params_linear_С = 0

total_iter_linear = len(C_vals)

i = 0
for C in C_vals:
    clear_output(wait=True)
    i += 1
    print(f'Current progress: {i}/{total_iter_linear}')
    model = svm.SVC(C=C, kernel='linear')
    model.fit(my_X_train, my_y_train.flatten())

    score = model.score(my_X_test, my_y_test)

    if score > my_best_score_linear:
        my_best_score_linear = score
        my_params_linear_С = C

print(f"Лучшая комбинация набрала {my_best_score_linear}: С = {my_params_linear_С}")

my_linear_best_c = 0.04
my_linear_model = svm.SVC(C=my_linear_best_c, kernel='linear')
my_linear_model.fit(my_X_train, my_y_train.flatten())
my_linear_model.score(my_X_test, my_y_test)

my_gaussian_model = svm.SVC(C=5, kernel='rbf', gamma=sigma_to_gamma_ang(10))
my_gaussian_model.fit(my_X_train, my_y_train.flatten())
my_gaussian_model.score(my_X_test, my_y_test)


def predict_text2(model, data, vocab):
    total = len(data)
    correct = 0
    for i in range(len(data)):
        item = data[i]
        email_vec = text_to_vec(item[0], vocab)
        y_pred = model.predict(email_vec)
        if (y_pred == item[1]): correct += 1
        # print(f"Text: #{i} - {item[1]}. Predicted: {y_pred}")

    accuracy = round(correct / total, 4)
    print('---------------------------')
    print(f'Accurecy: {accuracy * 100}%')


def predict_files(model, data, vocab):
    total = len(data)
    correct = 0
    for i in range(len(data)):
        item = data[i]
        email_vec = file_to_vec(item[0], vocab)
        y_pred = model.predict(email_vec)
        if (y_pred == item[1]): correct += 1
        # print(f"File: {item[0]} - {item[1]}. Predicted: {y_pred}")

    accuracy = round(correct / total, 4)
    print('---------------------------')
    print(f'Accurecy: {accuracy * 100}%')

print('Predict with Linear kernel')
predict_text2(my_linear_model, kaggle_emails, my_vocab)

print('Predict with gaussian kernel')
predict_text2(my_gaussian_model, kaggle_emails, my_vocab)

print('Predict with Linear kernel')
predict_files(my_linear_model, emails_data, my_vocab)

print('Predict with gaussian kernel')
predict_files(my_gaussian_model, emails_data, my_vocab)
