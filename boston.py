from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score

data_set = load_boston()

data = scale(data_set['data'])
target = data_set['target']

p_params = np.linspace(start=1, stop=10, num=200)
kf = KFold(n_splits=5, random_state=42, shuffle=True)

result = []

for p in p_params:
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)

    score = cross_val_score(estimator=regressor, cv=kf, X=data, y=target, scoring='neg_mean_squared_error')

    mean_value = np.mean(score)
    result.append(mean_value)

print(np.max(result))




