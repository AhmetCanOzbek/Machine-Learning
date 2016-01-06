__author__ = 'NishantNath'

# !/usr/bin/env python
'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py
Required packages : numpy, pandas, matplotlib, sklearn

Steps:
1.

# Uses Randomized PCA to find the most important features
# Uses Randomized PCA + Inverse Transform to find the most important features
'''

import pandas
import matplotlib.pyplot as mpyplot
import pylab
import numpy

# [0: 'CLASSICAL', 1: 'METAL', 2: 'HIPHOP', 3: 'DANCE', 4: 'JAZZ']
# [5:'FOLK', 6: 'SOUL', 7: 'ROCK', 8: 'POP', 9: 'BLUES']

col_input=['genre', 'year', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col22', 'col23', 'col24', 'col25', 'col26', 'col27', 'col28', 'col29', 'col30', 'col31', 'col32', 'col33', 'col34', 'col35', 'col36', 'col37', 'col38', 'col39', 'col40', 'col41', 'col42', 'col43', 'col44', 'col45', 'col46', 'col47', 'col48', 'col49', 'col50', 'col51', 'col52', 'col53', 'col54', 'col55', 'col56', 'col57', 'col58', 'col59', 'col60', 'col61', 'col62', 'col63', 'col64', 'col65', 'col66', 'col67', 'col68', 'col69', 'col70', 'col71', 'col72']
df_input = pandas.read_csv('pandas_output_missing_data_fixed.csv', header=None, delimiter = ",", names=col_input)

# range(2,74) means its goes from col 2 to col 73
df_input_data = df_input[list(range(2, 74))]
df_input_target = df_input[list(range(0, 1))]

colors = numpy.random.rand(len(df_input_target))

# Randomized PCA
from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(n_components=6) #from optimal pca components chart n_components=6
proj1 = pca.fit_transform(df_input_data)

# Relative weights on features
print pca.explained_variance_ratio_
print pca.components_

# Plotting
mpyplot.figure(1)
p1 = mpyplot.scatter(proj1[:, 0], proj1[:, 1], c=colors)
mpyplot.colorbar(p1)
mpyplot.show(p1)

# Randomized PCA using inverse transform - to make it linear
proj2 = pca.inverse_transform(proj1)

# Plotting
mpyplot.figure(2)
# p1 = mpyplot.scatter(proj1[:, 0], proj1[:, 1], c=colors, alpha=0.2)
p2 = mpyplot.scatter(proj2[:, 0], proj2[:, 1], c=colors, alpha=0.8)
mpyplot.colorbar(p1)
mpyplot.show(p2)