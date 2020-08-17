import pandas
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale

raw_data = pandas.read_csv('data_seta/wine.data')

raw_data.columns = ['Sort', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']

classes = raw_data['Sort']
signs_old = raw_data.drop('Sort', axis=1)
signs = scale(signs_old)
kf = KFold(n_splits=5, random_state=42, shuffle=True)

kNN = []

for k in range(1, 51):
    classifier = KNeighborsClassifier(n_neighbors=k)

    score = cross_val_score(estimator=classifier, cv=kf, scoring='accuracy', X=signs, y=classes)

    mean_value = np.mean(score)
    kNN.append(mean_value)

print(max(kNN))
print(kNN.index(max(kNN)) + 1)
# plt.plot(kNN)
# plt.show()


