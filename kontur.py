import pandas
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, r2_score
from sklearn.svm import SVC
import re
from sklearn.model_selection import GridSearchCV
import numpy as np


data = pandas.read_csv('./data_seta/kontur_train.csv', index_col='id')
data = data.fillna('')

tares = {}

for i, tare in enumerate(list(data['tare'].unique())):
    tares[tare] = i


def filter_not_valuable_words(phrase):
    words = phrase.split(' ')
    valuable_words = []

    for word in words:
        word = re.sub('[^a-я%/]', ' ', word)
        if len(word) > 2:
            valuable_words.append(word)
        else:
            if word == 'г' or word == 'гр':
                valuable_words.append('грамм')
            if '%' in word:
                valuable_words.append('процент')
            if word == 'л':
                valuable_words.append('литр')
            if word == 'ву':
                valuable_words.append('вакуумная')
            if word == 'вс':
                pass
            if 'жб' in word:
                valuable_words.append('банкаметаллическая')
    return ' '.join(valuable_words)


y_data = data.replace({'tare': tares})['tare'].to_numpy()

X_data = data['name'].str.lower().replace('[0-9/]', '', regex=True).map(filter_not_valuable_words)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
print('1')

kf = KFold(n_splits=5, random_state=42, shuffle=True)
param_grid = {
    'n_estimators': [20, 35, 40, 50],
    'bootstrap': [True],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

rfc = RandomForestClassifier(random_state=42, n_estimators=35)
print('2')
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=kf)
CV_rfc.fit(X_train, y_train)

print(CV_rfc.best_params_)

rfc1 = RandomForestClassifier(random_state=42, n_estimators=35, max_depth=8)
rfc1.fit(X_train, y_train)

X_test_1 = vectorizer.transform(X_test)

y_pred_forest = rfc.predict(X_test_1)

score = accuracy_score(y_test, y_pred_forest)
#
print(score)

clf = GradientBoostingClassifier(n_estimators=35, random_state=42)
clf.fit(X_train, y_train)

y_pred_grad = clf.predict(X_test_1)
score = accuracy_score(y_test, y_pred_grad)
#
print(score)