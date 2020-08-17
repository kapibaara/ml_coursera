import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

scaler = StandardScaler()

#train
train_data = pandas.read_csv('data_seta/percepton_train.csv')
y_train = train_data['class']
X_train = train_data.drop('class', axis=1)
print(y_train)
X_train_scaled = scaler.fit_transform(X_train)

#test
test_data = pandas.read_csv('data_seta/perceptron_test.csv')
y_test = test_data['class']
X_test = test_data.drop('class', axis=1)
X_test_scaled = scaler.transform(X_test)


clf = Perceptron(random_state=241, max_iter=5, tol=None)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)


clf_scale = Perceptron(random_state=241, max_iter=5, tol=None)
clf_scale.fit(X_train_scaled, y_train)
predictions_scaled = clf_scale.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, predictions_scaled)
print(accuracy_scaled)

print(round(accuracy_scaled - accuracy, 3))

