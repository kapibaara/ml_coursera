import pandas
import math
from sklearn.metrics import roc_auc_score

C = 10
k = 0.1


data = pandas.read_csv('venv/data-logistic.csv', header=None, names=['y', 'first', 'second'])
y = data['y'].to_numpy()
x1 = data['first'].to_numpy()
x2 = data['second'].to_numpy()

l = len(y)

def calc_w1(w1, w2):
    sum = 0
    print(w1, w2)

    for i in range(0, l):
        sum += y[i] * x1[i] * (1 - (1 / (1 + math.exp(-y[i] * (w1 * x1[i] + w2 * x2[i])))))

    return w1 + k * (1 / l) * sum - k * C * w1


def calc_w2(w1, w2):
    sum = 0

    for i in range(0, l):
        sum += y[i] * x2[i] * (1 - (1 / (1 + math.exp(-y[i] * (w1 * x1[i] + w2 * x2[i])))))

    return w2 + k * (1 / l) * sum - k * C * w2

def distance(w1_old, w1_new, w2_old, w2_new):
    return math.sqrt((w1_new - w1_old) ** 2 + (w2_new - w2_old) ** 2)

w1 = 0.1
w2 = 0.1

for i in range(10000):
    w1_new = calc_w1(w1, w2)
    w2_new = calc_w2(w1, w2)

    w1_old = w1
    w2_old = w2

    w1 = w1_new
    w2 = w2_new

    if distance(w1_old, w1_new, w2_old, w2_new) < 1e-5:
        print(i)
        break

y_predict = []

for i in range(0, l):
    ax = 1 / (1 + math.exp(-w1 * x1[i] - w2 * x2[i]))
    y_predict.append(ax)

roc_auc = roc_auc_score(y, y_predict)
print(roc_auc)





