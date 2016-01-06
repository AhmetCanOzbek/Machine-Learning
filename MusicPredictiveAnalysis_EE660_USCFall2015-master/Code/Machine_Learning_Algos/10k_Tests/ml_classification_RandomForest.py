__author__ = 'Arnav'

# !/usr/bin/env python
'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py
Required packages : numpy, pandas, matplotlib, sklearn

# Uses Random Forest for classification
'''

import pandas
import matplotlib.pyplot as plt
import numpy

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

    df_input = pandas.read_csv('C:/Users/Nishant/Documents/AllMyRepositories/projectEE660/zzz_scrapFiles/from preprocessing folder/pandas_merged_output_cleaned_None.csv',
                               header=None, delimiter="|", names=col_input)
    df_input = df_input.dropna()

    df_input = df_input[df_input['Year'] != 0]
    #df_input = df_input[df_input['Year'] != 0][df_input['Year'] < 1992]
    #df_input = df_input[df_input['Year'] != 0][df_input['Year'] >= 1992]
    # #print df_input.info()
    #df_input = df_input_data[df_input_data['col9']<1965]

    import reduce_dimensions
    df_input_data = df_input[list(range(1, 70))].as_matrix()
    df_input_target = df_input[list(range(0, 1))].as_matrix()
    print "Before reduction: ", df_input.shape
    df_input = reduce_dimensions.using_LDA(df_input_data, df_input_target, 2)
    print "After reduction: ", df_input.shape


    # splitting the data into training and testing sets
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_input_data, df_input_target.tolist())

    # Import the random forest package
    from sklearn.ensemble import RandomForestClassifier
    # Create the random forest object which will include all the parameters for the fit
    forest = RandomForestClassifier(n_estimators=500)
    # Fit the training data to the Survived labels and create the decision trees
    forest = forest.fit(X_train, numpy.ravel(y_train[:]))
    # Take the same decision trees and run it on the test data
    predicted = forest.predict(X_test)

    # Prediction Performance Measurement
    matches = (predicted == [item for sublist in y_test for item in sublist])
    print "Accuracy : ", (matches.sum() / float(len(matches)))

    # Feature importance calculation
    importances = forest.feature_importances_   # importance of each feature in order
    std = numpy.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)    # standard deviation of features in order

    indices = numpy.argsort(importances)[::-1]  # get indices of the sorted important features in desc. order

    # Print the feature ranking
    print("Feature ranking:")

    col_input_new=[]
    for f in range(X_train.shape[1]):
        print "%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]), col_input[indices[f]]
        col_input_new.append(col_input[indices[f]])

    # Plot the feature importances of the forest
    #plt, ax = plt.subplots()
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), col_input_new)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=60)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    """
width = 0.5
ind = np.arange(10)
rect = ax.bar(ind, counts, width, color='r')
ax.set_xlim(-width, len(ind)+width)
ax.set_ylabel('Number of Songs in dataset')
ax.set_title('Count of songs per genre in dataset')
ax.set_xticks(ind+width/2)
xtickNames = ax.set_xticklabels(labels)
plt.setp(xtickNames, rotation=45, fontsize=10)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height), ha='center', va='bottom')
autolabel(rect)
plt.show()
Chat Conversation End
"""

