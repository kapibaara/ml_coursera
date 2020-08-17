from sklearn.svm import SVC
import pandas
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

train_data = pandas.read_csv('data_seta/svm_data.csv', header=None, names=['class', 'first', 'second'])
y_train = train_data['class']
X_train = train_data.drop('class', axis=1)

clf = SVC(C=100000, random_state=241, kernel='linear')
clf.fit(X_train, y_train)

print(clf.support_)