from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
print(X.shape)
print(y.shape)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(C=100000, random_state=241, kernel='linear')
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)
C = gs.best_params_['C']

X_test = vectorizer.transform(newsgroups.data)
clf_opt = SVC(C=C, random_state=241, kernel='linear')
clf_opt.fit(X_test, y)

weights = abs(clf_opt.coef_.todense().A1)
weights = np.argsort(weights)[-10:]
feature_mapping = vectorizer.get_feature_names()

answer = []
for weight in weights:
    answer.append(feature_mapping[weight])
answer =['sci', 'keith', 'bible', 'religion', 'sky', 'moon', 'atheists', 'atheism', 'god', 'space']
print(sorted(answer))