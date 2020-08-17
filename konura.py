import pandas
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
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
X_train = X_train.tolist()
X_test = X_test.tolist()

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train).toarray()
print(X_train)

#
kf = KFold(n_splits=5, random_state=42, shuffle=True)
clf = GradientBoostingClassifier(n_estimators=35, random_state=42)
random_forest = RandomForestClassifier(random_state=42, n_estimators=35)

# gradient_scores = []
# forest_scores = []
# for train_index, test_index in kf.split(X_train):
#     X_train, X_t = X_train[train_index], X_train[test_index]
#     y_train, y_t = y_train[train_index], y_train[test_index]
#
#     clf.fit(X_train, y_train)
#     random_forest.fit(X_train, y_train)
#
#     gradient_predictions = clf.predict(X_t)
#     forest_predictions = clf.predict(X_t)
#
#     gradient_score = accuracy_score(y_t, gradient_predictions)
#     forest_score = accuracy_score(y_t, forest_predictions)
#     print(gradient_score, forest_score)
#
#     gradient_scores.append(gradient_score)
#     forest_scores.append(forest_score)
#
# print('градиент', np.mean(gradient_scores))
# print('лес', np.mean(forest_scores))
#
# ##
#
# X_test = vectorizer.transform(X_test).toarray()
# y_pred_grad = clf.predict(X_test)
# y_pred_forest = random_forest.predict(X_test)
#
# accuracy_grad = accuracy_score(y_test, y_pred_grad)
# accuracy_forest = accuracy_score(y_test, y_pred_forest)
# print(accuracy_grad)
# print(accuracy_forest)






