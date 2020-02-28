from pca import PCA
import numpy as np
import pandas as pd
from sklearn import preprocessing
import random

random.seed(1234)
np.random.seed(1234)


"""
load data
"""

col_names = [
    'target',
    'Alcohol',
    'Malicacid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280 /OD315 of diluted wines',
    'Proline'
    ]

data_path = 'data/wine.data'
data = pd.read_csv(data_path, sep=",")
data.columns = col_names
n = len(data)

# pick random sample
test_data = data.sample(1)
data = data.drop(test_data.index)

X_train = data.loc[:, 'Alcohol':]
y_train = data['target']

X_test = test_data.loc[:, 'Alcohol':]
y_test = test_data['target']

target_names = [str(i) for i in np.unique(y_train)]

print()
print('test data of class', y_test)

"""
scale
"""
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

"""
PCA
"""
pca = PCA()
pca.fit_transform(X_train)
pca.add_data(X_test)
plt = pca.plot(
    y_train, target_names, title='PCA of Wine dataset', plot_ellipse=True)
plt.savefig('output/pca_wine.png')
plt.show()
