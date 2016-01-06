# !/usr/bin/env python

__author__ = 'NishantNath'


'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py
Required packages : numpy, pandas, matplotlib, sklearn

Steps:
1.

# Uses nmf for prediction of artist hotness
'''

import numpy
import pandas

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

if __name__ == "__main__":

    # # [0: 'CLASSICAL', 1: 'METAL', 2: 'HIPHOP', 3: 'DANCE', 4: 'JAZZ']
    # # [5:'FOLK', 6: 'SOUL', 7: 'ROCK', 8: 'POP', 9: 'BLUES']
    #
    # col_input=['track_id', 'song_hotness', 'artist_hotness', 'artist_familiarity', 'mode', 'tempo', 'time-signature']
    # df_input = pandas.read_csv('somefilename.csv', header=None, delimiter = ",", names=col_input)
    #
    # # range(2,74) means its goes from col 2 to col 73
    # df_input_data = df_input[list(range(2,74))].as_matrix() # test with few good features as determined through PCA?
    # df_input_target = df_input[list(range(0,1))].as_matrix()

# ROCK TRACBWP128C7196948 0.704963316972 0.608849018571 0.840432421742 0 86.82 4
# ROCK TRADJAO128F146D76E 0.706430072087 0.576903815588 0.833967972053 1 116.282 4
# ROCK TRALYLQ128F42710D6 nan 0.52048957098 0.730246318613 1 146.06 4
# METAL TRBBQVL128F93421CA nan 0.264368000414 0.385430251168 1 124.969 7
# METAL TRBEOSC128F426AB23 0.849873124008 0.609262136002 0.909957522742 1 196.201 4
# METAL TRBHIPU12903CC380E 0.52273326078 0.598555411894 0.928936926189 1 178.427 4
#

    R = [
        [0.704963316972,0.608849018571,0.840432421742,0,86.82,4],
        [0.706430072087,0.576903815588,0.833967972053,1,116.282,4],
        [0.849873124008,0.609262136002,0.909957522742,1,196.201,4],
        [20,0.598555411894,0.928936926189,1,178.427,4]
        ]

    R = numpy.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K)
    nR = numpy.dot(nP, nQ.T)

    print nR

    # # design idea:
    # build a matrix of tracks in rows and features in col like artist hotness, song hotness, artist familiarity, major, minor, mode, tempo, time signature, etc
    # data to be esitmated needs to provides whatever is available & rest filled by nmf.