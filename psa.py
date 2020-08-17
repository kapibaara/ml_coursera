import pandas
from sklearn.decomposition import PCA
import numpy

data_train = pandas.read_csv('data_seta/close_prices.csv')
X_train = data_train.drop('date', axis=1)

pca = PCA(n_components=4)
pca.fit(X_train)
transform_data = pca.transform(X_train)
transform_data = list(transform_data.take([0], axis=1).flat)
print(numpy.argmax
(pca.components_[0]))

data_djia = pandas.read_csv('venv/djia_index.csv')
X_djia = data_djia['^DJI']

pirson = numpy.corrcoef(transform_data, X_djia)
print(round(pirson[0][1], 2))