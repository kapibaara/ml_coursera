import pandas
import numpy
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

data = pandas.read_csv('data_seta/gbm-data.csv')

y = data['Activity'].to_numpy()
X = data.drop('Activity', axis=1).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.8,
    random_state=241
)

learning_rate = [0.2]

def sigmoid(y):
    return 1 / (1 + math.exp(-y))

for rate in learning_rate:
    test_loss = []
    train_loss = []

    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=rate)
    clf.fit(X_train, y_train)

    y_pred_train = enumerate(clf.staged_decision_function(X_train))
    y_pred_test = enumerate(clf.staged_decision_function(X_test))

    for i, pred in y_pred_train:
        train_loss.append(log_loss(y_train, list(map(sigmoid, pred))))


    for i, pred in y_pred_test:
        test_loss.append(log_loss(y_test, list(map(sigmoid, pred))))

    print(round(numpy.min(test_loss), 2))
    print(numpy.argmin(test_loss))


    # plt.figure()
    # plt.plot(test_loss, 'r', linewidth=2)
    # plt.plot(train_loss, 'g', linewidth=2)
    # plt.legend(['train', 'test'])
    # plt.title('learning_rate: {}'.format(rate))
    # plt.show()


clf = RandomForestClassifier(n_estimators=36, verbose=True, random_state=241)
clf.fit(X_train, y_train)

pred = clf.predict_proba(X_test)

log_loss_value = log_loss(y_test, pred)

print(round(log_loss_value, 2))










