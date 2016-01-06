__author__ = 'NishantNath'

# !/usr/bin/env python
'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py
Required packages : numpy, pandas, matplotlib, sklearn

Steps:
1. Kernels : RBF (default), Linear, Poly, Precomputed, Sigmoid
2. Todo : try self-designed kernels?

# Uses SVM to find the most best clusters
'''

import pandas
import matplotlib.pyplot as mpyplot
import numpy
from mpl_toolkits import mplot3d
import pylab

# [0: 'CLASSICAL', 1: 'METAL', 2: 'HIPHOP', 3: 'DANCE', 4: 'JAZZ']
# [5:'FOLK', 6: 'SOUL', 7: 'ROCK', 8: 'POP', 9: 'BLUES']

def plot_3D(elev=30, azim=30):
    X = X_test
    y = numpy.random.rand(len(y_test))
    ax = mpyplot.subplot(projection='3d')
    r = numpy.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2))
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='spring')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

    return ax

# main function
if __name__ == '__main__':
    col_input=['genre', 'year', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col22', 'col23', 'col24', 'col25', 'col26', 'col27', 'col28', 'col29', 'col30', 'col31', 'col32', 'col33', 'col34', 'col35', 'col36', 'col37', 'col38', 'col39', 'col40', 'col41', 'col42', 'col43', 'col44', 'col45', 'col46', 'col47', 'col48', 'col49', 'col50', 'col51', 'col52', 'col53', 'col54', 'col55', 'col56', 'col57', 'col58', 'col59', 'col60', 'col61', 'col62', 'col63', 'col64', 'col65', 'col66', 'col67', 'col68', 'col69', 'col70', 'col71', 'col72']
    df_input = pandas.read_csv('pandas_output_missing_data_fixed.csv', header=None, delimiter = ",", names=col_input)

    # range(2,74) means its goes from col 2 to col 73
    df_input_data = df_input[list(range(2,74))].as_matrix()
    # to do: test with few good features as determined through PCA?
    df_input_target = df_input[list(range(0,1))].as_matrix()

    colors = numpy.random.rand(len(df_input_target))

    # splitting the data into training and testing sets
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_input_data, df_input_target.tolist())

    # SVM
    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=2),max_samples=0.5, max_features=0.5)
    bagging.fit(X_train[:],numpy.ravel(y_train[:]))
    predicted = bagging.predict(X_test)

    print y_test[60:90] , len(y_test[60:90])
    print predicted[60:90] , len(predicted[60:90])

    p1 = plot_3D()
    mpyplot.show(p1)

    # Prediction Performance Measurement
    matches = (predicted == [item for sublist in y_test for item in sublist])
    print matches.sum()
    print len(matches)

    print matches[10:50], len(matches[10:50])

    print "Accuracy : ", (matches.sum() / float(len(matches)))

    # # fix this metrics part later
    # from sklearn import metrics
    # print metrics.classification_report(y_test, predicted)
    #
    # print metrics.confusion_matrix(y_test, predicted)











