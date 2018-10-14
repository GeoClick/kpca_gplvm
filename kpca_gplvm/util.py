import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
import GPy

def convert_pca(X_train, X_test):
    """
    converts the data in X_test onto the first two principal components in X_train
    :param X_train: matrix in [num_points, num_dim]
    :param X_test: matrix in [num_points_test, num_dim]
    :return: matrix in [num_points_test, 2]
    """
    # Subtract the mean
    mean = np.mean(X_train, axis=0)
    X_train -= mean

    # Compute Singular Value Decomposition
    U, S, V = np.linalg.svd(X_train.T.dot(X_train))

    return (X_test - mean).dot(U[:, :2])


def convert_kpca(X_train, X_val):
    # Implements algorithm 14.2 from Murphy's book
    K_tt = np.exp(-1/2 * 1/0.05 * squareform(pdist(X_train, metric='sqeuclidean')))
    K_vt = np.exp(-1/2 * 1/0.05 * cdist(X_val, X_train, metric='sqeuclidean'))

    K_tt_cent = K_tt \
                - np.mean(K_tt, axis=1, keepdims=True)\
                - np.mean(K_tt, axis=0, keepdims=True) \
                + np.mean(np.mean(K_tt, axis=0, keepdims=True), axis=1, keepdims=True)
    K_vt_cent = K_vt \
                - np.mean(K_vt, axis=1, keepdims=True) \
                - np.mean(K_vt, axis=0, keepdims=True) \
                + np.mean(np.mean(K_vt, axis=0, keepdims=True), axis=1, keepdims=True)

    L, U = np.linalg.eig(K_tt_cent)

    V = U * np.sqrt(L)

    return K_vt_cent.dot(V[:, :2])


def convert_gplvm(X_train, X_val):
    m = GPy.models.GPLVM(X_train, input_dim=2, kernel=GPy.kern.RBF(input_dim=2, variance=0.05))
    m.optimize_restarts(optimizer='scg', num_restarts=2, messages=1, max_iters=3000)
    return m.X


def make_swiss_roll(num_points, noise_level=0.1, time_offset=0):
    """
    makes a swiss roll with a time_offset and noise added to X and Y coordinates
    :param num_points:
    :param noise_level:
    :param time_offset:
    :return:
    """
    # make the swiss roll noisy data set
    t = np.random.rand(num_points)  # Take random time points in [0, 1]

    # Add some noise to the X and Y coordinates
    X = t * np.cos(2 * np.pi * (t + time_offset)) + noise_level * np.random.randn(num_points)
    Y = t * np.sin(2 * np.pi * (t + time_offset)) + noise_level * np.random.randn(num_points)

    # The z coordinate is purposefully set at a smaller range
    Z = np.random.rand(num_points,) * 0.2
    return np.stack((X, Y, Z), axis=1)


def make_data_set(num_points):
    """
    Makes a data set of the swiss roll with phase offset between the groups
    :param num_points:
    :return:
    """
    X = np.concatenate((make_swiss_roll(num_points), make_swiss_roll(num_points, time_offset=0.5)), axis=0)
    Y = np.concatenate((np.zeros((num_points)), np.ones((num_points))), axis=0)
    return X, Y