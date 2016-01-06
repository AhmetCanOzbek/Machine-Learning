__author__ = "Can Ozbek"

import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def getUniqueCount(df_column):
    """
    Returns a dictionary of unique counts
    :param df_column: pandas series, (column)
    :return: dictionary containing unique counts
    """
    unique_values_list = df_column.unique().tolist()
    unique_count_dict = dict.fromkeys(unique_values_list)
    for value in unique_values_list:
        unique_count_dict[value] = sum(df_column == value)
    return unique_count_dict


def plot_confusion_matrix(y_true, y_predicted, title='Confusion matrix', cmap=plt.cm.GnBu):
    #Get unique class names
    classLabels = y_true.unique().tolist()
    #Get the confusion matrix
    cmatrix = confusion_matrix(y_true, y_predicted, classLabels)
    plt.imshow(cmatrix, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classLabels))
    plt.xticks(tick_marks, classLabels, rotation=45)
    plt.yticks(tick_marks, classLabels)
    for x in range(cmatrix.shape[0]):
        for y in range(cmatrix.shape[1]):
            plt.text(y,x,cmatrix[x][y],horizontalalignment='center',
                           verticalalignment='center')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cmatrix



df = pd.read_csv("/Users/ahmetcanozbek/Desktop/EE660/660Project/Extracted_Data/featureMatrix.bin",
                 header = 0, delimiter = "|")

#Getting the headers (Track ID, Song ID, ...)
#Returns a numpy array, to access the elements --> headers[0]
headers = df.columns.values

#Print out the info in original
print "***Original Dataset Shape: ", df.shape

#Clean the dataset
df_clean = df.dropna()
print "***Cleaned dataset Shape: ", df_clean.shape

#Read the Genre and Year True Labels
col_meta = ['Song ID', 'Track ID', 'Genre', 'Year']
df_meta = pd.read_csv('/Users/ahmetcanozbek/Desktop/EE660/660Project/Extracted_Data/msd_data_extract_2.bin',
                       header=None, delimiter = "|", names=col_meta)
#Delete Year From small file
df_meta = df_meta.drop('Year', axis=1)

#Merge the data and the Genre and Year labels
df_merged = pd.merge(df_clean, df_meta, how='left', on=['Track ID', 'Song ID'])

#Getting rid of UNCAT ones
df_merged = df_merged[df_merged["Genre"] != 'UNCAT']

#Start SVM for Genre Classification
print ""
print "*Start SVM Classification"
from sklearn.svm import SVC
svm_model = SVC(kernel = 'rbf', gamma = 1000)


y = df_merged["Genre"]
X = df_merged.drop(["Genre","Song ID","Track ID"], axis = 1)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
svm_model.fit(X_train, y_train)
y_train_predicted =  svm_model.predict(X_train)
y_test_predicted = svm_model.predict(X_test)

print "Number of Train Samples: ", (y_train.shape[0])
print "Number of Test Samples: ", (y_test.shape[0])

print "Train Classification Rate: ", (sum(y_train_predicted == y_train)) / float(y_train.shape[0])
print "Test Classification Rate: ", (sum(y_test_predicted == y_test)) / float(y_test.shape[0])

plt.figure()
plot_confusion_matrix(y_train,y_train_predicted,"C")
plt.show()


# cm_train = confusion_matrix(y_train, y_train_predicted, y.unique().tolist())
# print "Confusion Matrix Train: "
# print cm_train
# cm_test =  confusion_matrix(y_test, y_test_predicted, y.unique().tolist())
# print "Confusion Matrix Test: "
# print cm_test
# print y.unique().tolist()
# plt.figure()
# plot_confusion_matrix(cm_train,y.unique().tolist(),"CM train")
# plt.show()
# plt.figure()
# plot_confusion_matrix(cm_test,y.unique().tolist(),"CM test")
# plt.show()

# print "Type (df_input): ", type(df_input)
#
# df_input = df_input.dropna()
#
# df_input_data = df_input[range(1, 70)].as_matrix()
# df_input_target = df_input[list(range(0, 1))].as_matrix()
#
# # splitting the data into training and testing sets
# from sklearn.cross_validation import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(df_input_data, df_input_target.tolist())
#
# # Import the random forest package
# from sklearn.svm import SVC
# svc = SVC(probability=True, max_iter=10000, kernel='rbf') # kernel='rbf'
# # svc = SVC(kernel='linear', probability=True, max_iter=10000)
# # svc = SVC(kernel='poly', probability=True, max_iter=10000)
# # svc = SVC(kernel='precomputed', probability=True, max_iter=10000)
# # svc = SVC(kernel='sigmoid', probability=True, max_iter=10000) #results best for sigmoid
# svc.fit(X_train[:],numpy.ravel(y_train[:]))
# predicted = svc.predict(X_test)
#
# # Prediction Performance Measurement
# matches = (predicted == [item for sublist in y_test for item in sublist])
# print matches.sum()
# print len(matches)
#
# print matches[10:50], len(matches[10:50])
#
# print "Accuracy : ", (matches.sum() / float(len(matches)))


