import csv
import datetime
import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from kaggle import create_all_gold_feature, create_bag_of_words

data = pandas.read_csv('./data_seta/features.csv', index_col='match_id')
data = data.fillna(0)

heros = pandas.read_csv('../../Downloads/data/dictionaries/heroes.csv')
n_heros = len(heros['id'].unique())

y_data = data['radiant_win'].to_numpy()
kf = KFold(n_splits=6, random_state=42, shuffle=True)

columns = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero',
             'd4_hero', 'd5_hero', 'r1_gold', 'r2_gold', 'r3_gold', 'r4_gold', 'r5_gold', 'd1_gold', 'd2_gold', 'd3_gold',
             'd4_gold', 'd5_gold']

# фильтрация свойств, содержащих информацию, выходящую за пределы первых 5 минут матча
features = data.filter(columns)

# подсчет золота
X_data = create_all_gold_feature(features)
X_data = X_data.fillna({'r_all_gold': np.min(X_data['r_all_gold']), 'd_all_gold': np.min(X_data['d_all_gold'])})

# кодировка героев
X_heros = create_bag_of_words(features, n_heros)


# масштабирование

scaler = StandardScaler()

X_data = np.concatenate((X_heros, X_data), axis=1)
X_data = scaler.fit_transform(X_data)

clf = LogisticRegression(random_state=241, solver='lbfgs', class_weight='balanced').fit(X_data, y_data)
auc_score = cross_val_score(estimator=clf, cv=kf, scoring='roc_auc', X=X_data, y=y_data)

print(auc_score)

# тестовые данные
test_data = pandas.read_csv('./data_seta/features_test.csv', index_col='match_id')
test_data = test_data.fillna(0)

features_test = test_data.filter(columns)

X_data_test = create_all_gold_feature(features_test)
X_data_test = X_data_test.fillna({'r_all_gold': np.mean(X_data_test['r_all_gold']), 'd_all_gold': np.mean(X_data_test['d_all_gold'])})

X_heros_test = create_bag_of_words(features_test, n_heros)
X_data_test = np.concatenate((X_heros_test, X_data_test), axis=1)
X_data_test = scaler.transform(X_data_test)

pred_test = np.array(clf.predict_proba(X_data_test)[:, 1]).tolist()

data = pandas.read_csv('./data_seta/features_test.csv')
match_ids = data['match_id'].tolist()
#
# #
with open('./data_seta/output.csv', 'w', newline='') as csvfile:
    spamreader = csv.writer(csvfile, delimiter=',')

    spamreader.writerow(['match_id', 'radiant_win'])
    for i in range(len(match_ids)):
        spamreader.writerow([match_ids[i], pred_test[i]])





