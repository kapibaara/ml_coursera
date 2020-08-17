import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

data = pandas.read_csv('data_seta/abalone.csv')

X = data[
    ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight']].replace(
    {'M': 1, 'F': -1, 'I': 0})

print(X)

y = data['Rings'].to_numpy()
print(y)

kf = KFold(n_splits=5, random_state=1, shuffle=True)

for train_index, test_index in kf.split(X):
    print(train_index.shape)
    X_train, X_test = X[train_index], X[test_index]
    print(X_train.shape)
    y_train, y_test = y[train_index], y[test_index]

    regressor = RandomForestRegressor(random_state=1, n_estimators=5)
    regressor.fit(X_train, y_train)

    predictions = regressor.predict(X_test)

    score = r2_score(y_test, predictions)
