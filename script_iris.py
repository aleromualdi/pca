from sklearn import datasets
from pca import PCA


"""
load data
"""
iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

"""
pca
"""
pca = PCA()
pca.fit_transform(X)
plt = pca.plot(y, target_names, title='PCA of IRIS dataset', plot_ellipse=True)
plt.savefig('output/pca_iris.png')
