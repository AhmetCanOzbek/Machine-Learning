__author__ = 'NishantNath'

# !/usr/bin/env python
'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py
Required packages : numpy, pandas, matplotlib, sklearn

Steps:
1.

# Uses k-means (EM) for clustering
'''

import pandas
import matplotlib.pyplot as mpyplot
import pylab
import numpy
import sklearn
from scipy import stats
import seaborn as sns; sns.set()

# [0: 'CLASSICAL', 1: 'METAL', 2: 'HIPHOP', 3: 'DANCE', 4: 'JAZZ']
# [5:'FOLK', 6: 'SOUL', 7: 'ROCK', 8: 'POP', 9: 'BLUES']

col_input=['genre', 'year', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col22', 'col23', 'col24', 'col25', 'col26', 'col27', 'col28', 'col29', 'col30', 'col31', 'col32', 'col33', 'col34', 'col35', 'col36', 'col37', 'col38', 'col39', 'col40', 'col41', 'col42', 'col43', 'col44', 'col45', 'col46', 'col47', 'col48', 'col49', 'col50', 'col51', 'col52', 'col53', 'col54', 'col55', 'col56', 'col57', 'col58', 'col59', 'col60', 'col61', 'col62', 'col63', 'col64', 'col65', 'col66', 'col67', 'col68', 'col69', 'col70', 'col71', 'col72']
df_input = pandas.read_csv('pandas_output_missing_data_fixed.csv', header=None, delimiter = ",", names=col_input)

# range(2,74) means its goes from col 2 to col 73
df_input_data = df_input[list(range(2,74))].as_matrix() # test with few good features as determined through PCA?
df_input_target = df_input[list(range(0,1))].as_matrix()

colors = numpy.random.rand(len(df_input_target))

# splitting the data into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_input_data, df_input_target.tolist())

# k-means (EM)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)  # 4 clusters
kmeans.fit(X_train)
y_kmeans = kmeans.predict(X_test)
p1 = mpyplot.scatter(X_test[:, 0], X_test[:, 1], c=y_kmeans, s=50, cmap='rainbow')
mpyplot.show(p1)


# print y_test[10:50] , len(y_test[10:50])
# print predicted[10:50] , len(predicted[10:50])
#
# # Prediction Performance Measurement
# matches = (predicted == [item for sublist in y_test for item in sublist])
# print matches.sum()
# print len(matches)
#
# print matches[10:50], len(matches[10:50])
#
# print "Accuracy : ", (matches.sum() / float(len(matches)))
#
# # # fix this metrics part later
# # from sklearn import metrics
# # print metrics.classification_report(y_test, predicted)
# #
# # print metrics.confusion_matrix(y_test, predicted)

