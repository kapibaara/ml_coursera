import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas
import csv
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


data = pandas.read_csv('./data_seta/features.csv', index_col='match_id')
data = data.fillna(0)

# фильтрация свойств, содержащих информацию, выходящую за пределы первых 5 минут матча
features = data.drop(
    columns=['radiant_win', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant',
             'barracks_status_dire', 'duration'], axis=1)

kf = KFold(n_splits=6, random_state=42, shuffle=True)

X = features.to_numpy()
y = data['radiant_win'].to_numpy()


# кодировка героев (мешок слов)
heros = pandas.read_csv('../../Downloads/data/dictionaries/heroes.csv')
lobbies = pandas.read_csv('../../Downloads/data/dictionaries/lobbies.csv')
n_heros = len(heros['id'].unique())
n_lobbies = len(lobbies['id'].unique())


# pca = PCA(n_components=16)
# features_without_categorical_features = pca.fit_transform(features_without_categorical_features)


def create_bag_of_words(data, N):
    X_pick = np.zeros((data.shape[0], N))

    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.loc[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, data.loc[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    return X_pick


def create_bag_of_words_for_lobbies(data, N):
    X_pick = np.zeros((data.shape[0], N))

    for i, match_id in enumerate(data.index):
        for n in range(N):
            if data.loc[match_id, 'lobby_type'] == n - 1:
                X_pick[i, n] = 1
            else:
                X_pick[i, n] = -1
    return X_pick


def create_all_gold_feature(data):
    col_list = [['r1_gold', 'r2_gold', 'r3_gold', 'r4_gold', 'r5_gold'],
                ['d1_gold', 'd2_gold', 'd3_gold', 'd4_gold', 'd5_gold']]
    r_gold = []
    d_gold = []

    for i, match_id in enumerate(data.index):
        r_sum = 0
        d_sum = 0
        for p in range(5):
            r_sum += data.loc[match_id, 'r%d_gold' % (p + 1)]
            d_sum += data.loc[match_id, 'd%d_gold' % (p + 1)]
        r_gold.append(r_sum)
        d_gold.append(d_sum)

    d = {'r_all_gold': r_gold, 'd_all_gold': d_gold}
    df = pandas.DataFrame(data=d)
    return data.join(df)

columns = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
             'd4_hero', 'd5_hero', 'lobby_type', 'r1_lh', 'r2_lh', 'r3_lh',
           'r4_lh', 'r5_lh', 'd1_lh', 'd2_lh', 'd3_lh', 'd4_lh', 'd5_lh',
            'r1_level', 'r2_level', 'r3_level',
           'r4_level', 'r5_level', 'd1_level', 'd2_level', 'd3_level', 'd4_level', 'd5_level',
           ]

scaler = StandardScaler()

X_lobbies = create_bag_of_words_for_lobbies(features, n_lobbies)
X_gold = create_all_gold_feature(features)
X_gold = X_gold.fillna({'r_all_gold': np.mean(X_gold['r_all_gold']), 'd_all_gold': np.mean(X_gold['d_all_gold'])})
X_gold = X_gold.fillna({'r_all_gold': 0, 'd_all_gold': 0})

# убираем категориальные признаки
features_without_categorical_features = X_gold.drop(
    columns=columns, axis=1)

X_pick = create_bag_of_words(features, n_heros)
X_without_categorical_features = scaler.fit_transform(features_without_categorical_features)

X_with_bag_of_words = np.concatenate((X_pick, X_without_categorical_features, X_lobbies), axis=1)

y_data = data['radiant_win'].to_numpy()

clf = LogisticRegression(random_state=241, C=0.1, solver='lbfgs', class_weight='balanced').fit(X_with_bag_of_words,
                                                                                               y_data)
auc_score = cross_val_score(estimator=clf, cv=kf, scoring='roc_auc', X=X_with_bag_of_words, y=y_data)

print(clf.coef_[0])

print(auc_score)

# # проверка лучшей  модели на тестовой выборке (логистич регрессия с мешком слов)
# test_data = pandas.read_csv('./data_seta/features_test.csv', index_col='match_id')
# test_data = test_data.fillna(0)
#
# X_gold = create_all_gold_feature(test_data)
# X_gold = X_gold.fillna({'r_all_gold': np.mean(X_gold['r_all_gold']), 'd_all_gold': np.mean(X_gold['d_all_gold'])})
# X_gold = X_gold.fillna({'r_all_gold': 0, 'd_all_gold': 0})
#
# X_pick_test = create_bag_of_words(test_data, n_heros)
# X_lobbies = create_bag_of_words_for_lobbies(test_data, n_lobbies)
#
# X_without_categorical_features_test = X_gold.drop(
#     columns=columns, axis=1)
#
# # X_without_categorical_features_test = pca.transform(X_without_categorical_features_test)
#
# X_scaled_test = scaler.transform(X_without_categorical_features_test)
#
# X_with_bag_of_words_test = np.concatenate((X_pick_test, X_scaled_test, X_lobbies), axis=1)
#
# pred_test = np.array(clf.predict_proba(X_with_bag_of_words_test)[:, 1]).tolist()
#
# data = pandas.read_csv('./data_seta/features_test.csv')
# match_ids = data['match_id'].tolist()
# #
# # #
#
# with open('./data_seta/output.csv', 'w', newline='') as csvfile:
#     spamreader = csv.writer(csvfile, delimiter=',')
#
#     spamreader.writerow(['match_id', 'radiant_win'])
#     for i in range(len(match_ids)):
#         y_pred = 1 if pred_test[i] > 0.5 else 0
#         spamreader.writerow([match_ids[i], pred_test[i]])
