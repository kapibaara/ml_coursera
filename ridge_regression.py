import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

data = pandas.read_csv('/Users/a.boldovskaya/Downloads/salary-train.csv')

data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True).str.lower()
data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)
y_train = data['SalaryNormalized']

vectorizer = TfidfVectorizer(min_df=5)
X_train = vectorizer.fit_transform(data['FullDescription'])

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))

matrix_train = hstack([X_train_categ, X_train])

ridge = Ridge(alpha=1, random_state=241)
ridge.fit(matrix_train, y_train)

data_test = pandas.read_csv('data_seta/salary-mini.csv')
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True).str.lower()
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

X_test = vectorizer.transform(data_test['FullDescription'])
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

matrix_test = hstack([X_test_categ, X_test])
ridge.predict(matrix_test)

print(ridge.predict(matrix_test))
