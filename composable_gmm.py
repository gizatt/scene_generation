import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


def plot_ellipses(ax, weights, means, covars, facecolor="#56B4E9"):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(means[n], eig_vals[0], eig_vals[1],
                                  180 + angle, edgecolor='black')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor(facecolor)
        ax.add_artist(ell)


class ComposableGMM(GaussianMixture):
    def concat(self, y):
        ''' Returns a mixture that has all mixtures
            from both self and y concatenated. '''
        gmm = copy.deepcopy(self)
        if self.means_.shape[1] != y.means_.shape[1]:
            raise ValueError("ComposableGMM + ComposableGMM "
                             "data dimension mismatch.")
        if self.covariance_type != y.covariance_type:
            raise ValueError("ComposableGMM * ComposableGMM "
                             "have different covariance types.")
        # Combine means + weights by pure concatenation
        gmm.means_ = np.vstack((self.means_, y.means_))
        gmm.weights_ = np.hstack((self.weights_, y.weights_))
        # Renormalize the weights.
        gmm.weights_ /= np.sum(gmm.weights_)
        n_features = gmm.weights_.shape[0]

        # Updating covariances depends on convariance type:
        if self.covariance_type is 'full':
            # For full covariances, we can just stack them.
            gmm.covariances_ = np.vstack((self.covariances_,
                                          y.covariances_))
            gmm.precisions_ = np.vstack((self.precisions_,
                                         y.precisions_))
            gmm.precisions_cholesky_ = np.vstack(
                (self.precisions_cholesky_, y.precisions_cholesky_))
        else:
            raise NotImplementedError(
                "Covariance combination for non-full covariance "
                "not implemented.")
        return gmm

    def transform(self, T, R):
        ''' T should be n_features x 1.
            R should be n_features x n_features.
            Changes each mixture's mean location by R*<mean> + T.
            Returns the new mixture model. '''
        if not np.allclose(np.linalg.det(R), 1.):
            print "Warning: transforming ComposableGMM with non-affine" \
                  " TF (det %f). Covariances will be wrong." % np.linalg.det(R)
        gmm = copy.deepcopy(self)
        n_features = self.means_.shape[1]
        if (n_features != T.shape[0] or
           n_features != R.shape[0] or
           n_features != R.shape[1]):
            raise ValueError("ComposableGMM TF matrix dimension mismatch."
                             "Wanted %fx1 and %fx%f, got %s and %s." %
                             (str(T.shape), str(R.shape)))
        # Combine means + weights by pure concatenation
        for i in range(self.means_.shape[0]):
            gmm.means_[i, :] = R.dot(gmm.means_[i, :]) + T
        return gmm

    def prod(self, y, min_weight=0.0, max_components=None):
        ''' Combines the two mixtures by creating a new GMM with one
        Gaussian component for every pair of components across the two
        input mixtures, effectively multiplying them. Prunes lowest-weight
        product mixtures, if requested. '''
        gmm = copy.deepcopy(self)
        n_features = self.means_.shape[1]
        if n_features != y.means_.shape[1]:
            raise ValueError("ComposableGMM * ComposableGMM "
                             "have different # of features.")
        if self.covariance_type != y.covariance_type:
            raise ValueError("ComposableGMM * ComposableGMM "
                             "have different covariance types.")
        if self.covariance_type is not 'full':
            raise ValueError("Covariance types that aren't full aren't"
                             " supported yet.")
        n_components_1 = self.means_.shape[0]
        n_components_2 = y.means_.shape[0]

        # Go through and compute weights first, to prune by weights easier
        gmm.weights_ = np.empty(n_components_1*n_components_2)
        k = 0
        for i in range(n_components_1):
            for j in range(n_components_2):
                mu_1 = self.means_[i, :]
                mu_2 = y.means_[j, :]
                # Extracting covariances depends on convariance type:
                prec_1 = self.precisions_[i, :, :]
                prec_2 = y.precisions_[j, :, :]

                scale_factor = multivariate_normal.pdf(
                    mu_2, mean=mu_2, cov=prec_1 + prec_2)
                gmm.weights_[k] = (
                    self.weights_[i] * y.weights_[j] * scale_factor)
                k += 1
        gmm.weights_ /= np.sum(gmm.weights_)

        keep_inds = gmm.weights_ > min_weight
        if max_components is not None:
            # Get the max_components largest weights
            smallest_weight_inds = gmm.weights_.argsort()[:-max_components]
            keep_inds[smallest_weight_inds] = False
        num_keep = np.sum(keep_inds)

        # Compute everything else.
        gmm.weights_ = gmm.weights_[keep_inds]
        gmm.weights_ /= np.sum(gmm.weights_)
        gmm.means_ = np.empty((num_keep, n_features))
        gmm.covariances_ = np.empty((num_keep, n_features, n_features))
        gmm.precisions_ = np.empty((num_keep, n_features, n_features))
        gmm.precisions_cholesky_ = np.empty((num_keep, n_features, n_features))
        k = 0
        for i in range(n_components_1):
            for j in range(n_components_2):
                if not keep_inds[i*n_components_2+j]:
                    continue
                mu_1 = self.means_[i, :]
                mu_2 = y.means_[j, :]
                # Extracting covariances depends on convariance type:
                prec_1 = self.precisions_[i, :, :]
                prec_2 = y.precisions_[j, :, :]

                # Referencing
                # http://compbio.fmph.uniba.sk/vyuka/ml/old/2008/handouts/matrix-cookbook.pdf
                # page 41 (8.1.8) for product of two gaussians
                gmm.precisions_[k, :, :] = prec_1 + prec_2
                gmm.covariances_[k, :, :] = np.linalg.inv(
                    gmm.precisions_[k, :, :])
                gmm.precisions_cholesky_[k, :, :] = np.linalg.cholesky(
                    gmm.precisions_[k, :, :])
                gmm.means_[k, :] = gmm.covariances_[k, :, :].dot(
                    prec_1.dot(mu_1) + prec_2.dot(mu_2))

                k += 1

        return gmm

    def __add__(self, y):
        ''' If y is a float or a numpy vector, shifts the means of the
            GMM by that amount.
            If y is a ComposableGMM or a BayesianGaussianMixture,
            combines the two mixtures by concatenating them.'''
        if issubclass(type(y), GaussianMixture) or \
           issubclass(type(y), BayesianGaussianMixture):
            return self.concat(y)
        else:
            raise ValueError("Unsupported addition: ComposableGMM + %s" %
                             y.__class__.__name__)

    def __mul__(self, y):
        ''' Combines the two mixtures by creating a new GMM with one
        Gaussian component for every pair of components across the two
        input mixtures, effectively multiplying them. '''
        if issubclass(type(y), GaussianMixture) or \
           issubclass(type(y), BayesianGaussianMixture):
            return self.prod(y, min_weight=0., max_components=None)
        else:
            raise ValueError("Unsupported multiplication: ComposableGMM + %s" %
                             y.__class__.__name__)

    def plot(self, ax, **kwargs):
        plot_ellipses(ax, self.weights_, self.means_, self.covariances_, **kwargs)


if __name__ == "__main__":
    test_data_1 = np.random.random((1000, 2)) * 0.5
    test_data_2 = np.random.random((500, 2))
    test_data_2[:, 0] += 1.

    gmm_1 = ComposableGMM(n_components=2)
    gmm_1.fit(test_data_1)
    gmm_2 = ComposableGMM(n_components=2)
    gmm_2.fit(test_data_2)

    gmm_add = gmm_1 + gmm_2
    gmm_mul = gmm_1.prod(gmm_2, max_components=2)

    print gmm_add.means_, gmm_add.weights_
    print gmm_mul.means_, gmm_mul.weights_
    fig, ((ax_1, ax_2), (ax_add, ax_mul)) = plt.subplots(2, 2)
    ax_1.scatter(test_data_1[:, 0], test_data_1[:, 1])
    gmm_1.plot(ax_1)
    ax_2.scatter(test_data_2[:, 0], test_data_2[:, 1])
    gmm_2.plot(ax_2)

    ax_add.scatter(test_data_1[:, 0], test_data_1[:, 1])
    ax_add.scatter(test_data_2[:, 0], test_data_2[:, 1])
    gmm_add.plot(ax_add)

    ax_mul.scatter(test_data_1[:, 0], test_data_1[:, 1])
    ax_mul.scatter(test_data_2[:, 0], test_data_2[:, 1])
    gmm_mul.plot(ax_mul)

    plt.show()