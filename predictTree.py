import numpy
import pandas
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('data_seta/titanic.csv', usecols=['Sex', 'Age', 'Pclass', 'Fare', 'Survived'])

filtered_data = data.dropna()
traning_sample = filtered_data[['Sex', 'Age', 'Pclass', 'Fare']].replace({'male': 0, 'female': 1}).to_numpy()

survived = filtered_data['Survived'].to_numpy()
kf = KFold(n_splits=6, random_state=42, shuffle=True)

clf = DecisionTreeClassifier()

score = cross_val_score(estimator=clf, cv=kf, scoring='accuracy', X=traning_sample, y=survived)

print(numpy.mean(score))
