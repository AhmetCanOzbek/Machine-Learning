__author__ = 'Arnav'

"""
This module provides fuctionality for reducing dimentions of data set by
decomposing a multivariate dataset in a set of successive orthogonal components
that explain a maximum amount of the variance.
"""

def using_PCA(feature_mat, reduced_dim):
    """
    Linear dimensionality reduction using Singular Value Decomposition
    of the data and keeping only the most significant singular vectors
    :param featureMat: numpy array of float/int features with no missing values
    :param reduced_dim: required number of dimensions
    :return: feature matrix with reduced_dim number of dimensions
    """
    from sklearn.decomposition import PCA
    reduced_feature_mat = PCA(n_components=reduced_dim).fit(feature_mat).transform(feature_mat)
    return reduced_feature_mat

def using_LDA(feature_mat, true_labels, reduced_dim):
    """
    Reduces the dimensionality of the input by projecting it to the most discriminative directions.
    :param features_mat: numpy array of float/int features with no missing values
    :param true_labels: numpy array of true labels
    :param reduced_dim: required number of dimensions
    :return: reduced feature matrix with reduced_dim number of dimensions
    """
    from sklearn.lda import LDA
    import numpy
    reduced_feature_mat = LDA(n_components=reduced_dim).fit(feature_mat, numpy.ravel(true_labels[:])).transform(feature_mat)
    return reduced_feature_mat
