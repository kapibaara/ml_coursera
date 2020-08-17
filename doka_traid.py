import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

data = pandas.read_csv('./data_seta/features.csv', index_col='match_id')
data = data.fillna(0)

# фильтрация свойств, содержащих информацию, выходящую за пределы первых 5 минут матча
features = data.drop(
    columns=['radiant_win', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant',
             'barracks_status_dire'])

features_with_nan = features.columns[features.isna().any()].tolist()

kf = KFold(n_splits=5, random_state=42, shuffle=True)


# ------------------Gradient Boosting------------------

X = features.to_numpy()
y = data['radiant_win'].to_numpy()

# оптимум достигается при 100, если не задавать глубину
# при задании глубины 5, оптимум достигается при 30
estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
auc_scores = []

for estimator in estimators:
    print('-----------{}-----------'.format(estimator))
    scores = []

    start_time = datetime.datetime.now()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = GradientBoostingClassifier(n_estimators=estimator, random_state=241)

        clf.fit(X_train, y_train)

        pred = clf.predict_proba(X_test)[:, 1]

        scores.append(roc_auc_score(y_test, pred))
    print('Time elapsed:', datetime.datetime.now() - start_time)

    auc_score = np.mean(scores)
    auc_scores.append(auc_score)
    print(auc_score)

plt.plot(estimators, auc_scores)
plt.ylabel('auc scores')
plt.xlabel('n estimators')
plt.grid()
plt.show()

# ------------------Log Regression------------------


def logistic_regression(X_data, y_data, title):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_data)

    auc_scores = []
    C_params = np.linspace(start=0.1, stop=1, num=10)

    for C in C_params:
        print('-----------{}-----------'.format(C))
        scores = []

        start_time = datetime.datetime.now()

        clf = LogisticRegression(random_state=0, C=C)

        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            clf.fit(X_train, y_train)
            pred = clf.predict_proba(X_test)[:, 1]

            scores.append(roc_auc_score(y_test, pred))
        auc_score = np.mean(scores)
        auc_scores.append(auc_score)

        print('Time elapsed:', datetime.datetime.now() - start_time)

    print(np.max(auc_scores))

    plt.title(title)
    plt.plot(C_params, auc_scores)
    plt.ylabel('auc scores')
    plt.xlabel('C')
    plt.grid()
    plt.show()


y = data['radiant_win'].to_numpy()

logistic_regression(features, y, 'with categorical features')



# убираем категориальные признаки
features_without_categorical_features = features.drop(
     columns=['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
             'd4_hero', 'd5_hero'])

logistic_regression(features_without_categorical_features, y, 'without categorical features')



# кодировка героев (мешок слов)
columns = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
           'd4_hero', 'd5_hero']

heros = pandas.read_csv('../../Downloads/data/dictionaries/heroes.csv')
n_heros = len(heros['id'].unique())


def create_bag_of_words(data, N):
    X_pick = np.zeros((data.shape[0], N))

    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.loc[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, data.loc[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    return X_pick


X_pick = create_bag_of_words(features, n_heros)
X_without_categorical_features = features_without_categorical_features.to_numpy()

X_with_bag_of_words = np.concatenate((X_pick, X_without_categorical_features), axis=1)
y_data = data['radiant_win'].to_numpy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_with_bag_of_words)

auc_scores = []

start_time = datetime.datetime.now()

clf = LogisticRegression(random_state=0, C=0.1)

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]

    auc_scores.append(roc_auc_score(y_test, pred))

auc_score = np.mean(auc_scores)

print(np.max(auc_scores))



# проверка лучшей  модели на тестовой выборке (логистич регрессия с мешком слов)
test_data = pandas.read_csv('./data_seta/features_test.csv', index_col='match_id')
test_data = test_data.fillna(0)

X_pick_test = create_bag_of_words(test_data, n_heros)
X_without_categorical_features_test = test_data.drop(
     columns=['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
             'd4_hero', 'd5_hero'])

X_with_bag_of_words_test = np.concatenate((X_pick, X_without_categorical_features), axis=1)

X_scaled_test = scaler.transform(X_with_bag_of_words_test)
pred_test = clf.predict_proba(X_scaled_test)[:, 1]

print(np.max(pred_test), np.min(pred_test))



