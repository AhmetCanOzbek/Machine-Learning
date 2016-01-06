__author__ = 'Arnav'

# !/usr/bin/env python
'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py
Required packages : numpy, pandas, sklearn

# Uses QDA for classification
'''

import pandas
import numpy as np

# main function
if __name__ == '__main__':
    col_input = ['genre', 'AvgBarDuration','Loudness', 'Tempo','ArtistFamiliarity','ArtistHotttnesss','SongHotttnesss',
                 'Mode[0]','Mode[1]','Year',

                 'Key[0]','Key[1]','Key[2]','Key[3]','Key[4]','Key[5]',
                 'Key[6]','Key[7]','Key[8]','Key[9]','Key[10]','Key[11]',

                 'PicthesMean[0]','PicthesMean[1]','PicthesMean[2]','PicthesMean[3]','PicthesMean[4]','PicthesMean[5]',
                 'PicthesMean[6]','PicthesMean[7]','PicthesMean[8]','PicthesMean[9]','PicthesMean[10]','PicthesMean[11]',

                 'PitchesVar[0]','PitchesVar[1]','PitchesVar[2]','PitchesVar[3]','PitchesVar[4]','PitchesVar[5]',
                 'PitchesVar[6]','PitchesVar[7]','PitchesVar[8]','PitchesVar[9]','PitchesVar[10]','PitchesVar[11]',

                 'TimbreMean[0]','TimbreMean[1]','TimbreMean[2]','TimbreMean[3]','TimbreMean[4]','TimbreMean[5]',
                 'TimbreMean[6]','TimbreMean[7]','TimbreMean[8]','TimbreMean[9]','TimbreMean[10]','TimbreMean[11]',

                 'TimbreVar[0]','TimbreVar[1]','TimbreVar[2]','TimbreVar[3]','TimbreVar[4]','TimbreVar[5]',
                 'TimbreVar[6]','TimbreVar[7]','TimbreVar[8]','TimbreVar[9]','TimbreVar[10]','TimbreVar[11]']

    df_input = pandas.read_csv('pandas_merged_output_cleaned_None.csv',
                               header=None, delimiter="|", names=col_input)
    df_input = df_input.dropna()

    #df_input = df_input[df_input['Year'] != 0][df_input['genre'] != 'CLASSICAL']
    #df_input = df_input[df_input['Year'] != 0][df_input['Year'] < 1992][df_input['genre'] != 'CLASSICAL']
    df_input = df_input[df_input['Year'] != 0][df_input['Year'] >= 1992][df_input['genre'] != 'CLASSICAL']

    df_input_target = df_input[list(range(0, 1))].as_matrix()
    df_input_data = df_input[list(range(1, 70))].as_matrix()

    # splitting the data into training and testing sets
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_input_data, df_input_target.tolist())

    # Start QDA Classification
    from sklearn.qda import QDA
    clf = QDA(priors=None, reg_param=0.001).fit(X_train, np.ravel(y_train[:]))
    predicted = clf.predict(X_test)
    matches = (predicted == [item for sublist in y_test for item in sublist])
    print "Accuracy : ", (matches.sum() / float(len(matches)))

