import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.decomposition import PCA as sklearnPCA


class PCA(object):
    """
    This class implements a wrapper around the scikit-learn class
    sklearn.decomposition.PCA, performing first 2 componen analysis.
    The class includes plotting functionalities to visualize the multivariate
    model.

    """

    def __init__(self):
        """ c'tor """

        self.pca = sklearnPCA(n_components=2)
        self.fit_executed = False
        self.X_tr = None
        self.Xn_tr = None

    def _check_fit_executed(self):
        if not self.fit_executed:
            raise RuntimeError('PCA model not fit yet. Please fit data first.')

    def fit_transform(self, X):
        """
        Wrapper to sci-kit learn method 'fit' the model with X.


        :param X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        :return : returns the instance itself.
        """

        self.X_tr = self.pca.fit(X).transform(X)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        self.fit_executed = True

    @staticmethod
    def _get_covariance_ellipse(cdata_0, cdata_1, nstd=2, **kwargs):
        """
        Method to calculate the covariance ellipsis, that  allows to visualize
        the nstd * confidence interval within which the PCA data are
        distributed.

        :param cdata_0 :
        :param cdata_1 :
        :param nstd :

        :return : a matplotlib Ellipse patch representing the covariance matrix
            cov centred at centre and scaled by the factor nstd.
        """

        cov = np.cov(cdata_0, cdata_1)
        centre = (np.mean(cdata_0), np.mean(cdata_1))

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
        theta = np.arctan2(vy, vx)

        width, height = 2 * nstd * np.sqrt(eigvals)
        return Ellipse(xy=centre, width=width, height=height,
                       angle=np.degrees(theta), **kwargs)

    def add_data(self, data):
        """
        Add new data to the PCA

        """
        self._check_fit_executed()
        self.Xn_tr = self.pca.transform(data)

    def plot(self, y=None, target_names=None, title='',
            plot_ellipse=False,
            *args, **kwargs):
        """
        Plotting method.

        :param y : (default=None)
        :param target_names : (default=None)
        :param title: str, title
        :param plot_ellipse: bool (default=False),
            if True plot covariance ellipse
        :return : matplotlib.pyplot instance

        """

        del args
        del kwargs

        self._check_fit_executed()

        evr = self.explained_variance_ratio

        fig, ax = plt.subplots()

        palette = itertools.cycle(sns.color_palette())

        for i, trg_name in zip((np.unique(y)), target_names):
            color = next(palette)
            cdata = self.X_tr[y == i, :]
            cdata_0 = cdata[:, 0]
            cdata_1 = cdata[:, 1]

            ax.scatter(
                cdata_0,
                cdata_1,
                color=color,
                alpha=.8,
                lw=1,
                label=trg_name)

            if plot_ellipse:
                ell = self._get_covariance_ellipse(
                    cdata_0, cdata_1, fc=color, alpha=0.4)
                ax.add_artist(ell)

        if self.Xn_tr is not None:
            ax.scatter(
                self.Xn_tr[:, 0],
                self.Xn_tr[:, 1],
                color='k',
                # alpha=.8,
                lw=1,
                label='new_data')

        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.xlabel('PC$^0$ [{0:.2f} %]'.format(evr[0]))
        plt.ylabel('PC$^1$ [{0:.2f} %]'.format(evr[1]))
        plt.title(title)
        return plt
