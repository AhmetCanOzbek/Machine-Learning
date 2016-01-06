__author__ = "Can Ozbek Arnav"

import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import sys
sys.path.append("/Users/ahmetcanozbek/Desktop/EE660/660Project/Code_Final_Used/functions")
import ml_aux_functions as ml_aux
import crop_rock


#PREPROCESSING
#Read the files
df_full = pd.read_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t3.pkl")   # 80%
print "DEBUG: file read."

#Get rid of the rows that have missing values (nan) and UNCAT
df_full = df_full[ df_full["Genre"] != "UNCAT" ]
df_full = df_full.dropna()
y_full = df_full["Genre"]
X_full = df_full.drop(["Genre", "Track ID", "Year"], axis=1)

#Split the 80% of data to 70% Training and 30% Validation Data
from sklearn.cross_validation import train_test_split
X_train, X_validation, y_train, y_validation = \
                            train_test_split(X_full, y_full, train_size=0.7, random_state=42)
print "DEBUG: Data splitted"
df_train_toCrop = pd.concat([y_train, X_train], axis=1, join='inner')

#Crop the dataset
maxval = crop_rock.find_second_max_value(df_train_toCrop)
df_cropped = crop_rock.drop_excess_rows(df_train_toCrop, maxval)
y_cropped = df_cropped["Genre"]
X_cropped = df_cropped.drop(["Genre"], axis=1)

# Start LDA Classification
print "Performing LDA Classification:"
from sklearn.lda import LDA
clf = LDA(solver='svd', shrinkage=None, n_components=None).fit(X_cropped, np.ravel(y_cropped[:]))

#Use X_cropped to get best model
y_train_predicted = clf.predict(X_train)
print "Error rate for LDA on Training: ", ml_aux.get_error_rate(y_train,y_train_predicted)
# ml_aux.plot_confusion_matrix(y_cropped, predicted, "CM on LDA cropped")
# plt.show()

y_validation_predicted = clf.predict(X_validation)
print "Error rate for LDA on Validation: ", ml_aux.get_error_rate(y_validation,y_validation_predicted)
# ml_aux.plot_confusion_matrix(y_validation, y_validation_predicted, "CM on LDA validation (t1)")
# plt.show()



# Start Adaboost Classification
from sklearn.ensemble import AdaBoostClassifier
adaboost_model = AdaBoostClassifier(n_estimators=50)
adaboost_model = adaboost_model.fit(X_cropped,y_cropped)
# predicted = adaboost_model.predict(X_cropped)
# print "Error rate for LDA on Cropped: ", ml_aux.get_error_rate(y_cropped,predicted)
# ml_aux.plot_confusion_matrix(y_cropped, predicted, "CM on LDA cropped")
# plt.show()

y_validation_predicted = adaboost_model.predict(X_validation)
print "Error rate for Adaboost on Validation: ", ml_aux.get_error_rate(y_validation,y_validation_predicted)
# ml_aux.plot_confusion_matrix(y_validation, y_validation_predicted, "CM on Adaboost validation (t1)")
# plt.show()


# Start QDA Classification
print "Performing QDA Classification:"
from sklearn.qda import QDA
clf = QDA(priors=None, reg_param=0.001).fit(X_cropped, np.ravel(y_cropped[:]))
y_validation_predicted = clf.predict(X_validation)
print "Error rate for QDA (Validation): ", ml_aux.get_error_rate(y_validation,y_validation_predicted)



# Start Random Forest Classification
print "Performing Random Classification:"
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=500)
forest = forest.fit(X_cropped, np.ravel(y_cropped[:]))
y_validation_predicted = forest.predict(X_validation)
print "Error rate for Random Forest (Validation): ", ml_aux.get_error_rate(y_validation,y_validation_predicted)
# ml_aux.plot_confusion_matrix(y_validation, y_validation_predicted, "CM Random Forest (t1)")
# plt.show()


# Start k nearest neighbor Classification
print "Performing kNN Classification:"
from sklearn import neighbors
knn_model = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='auto',leaf_size=15)
knn_model.fit(X_cropped, y_cropped)
# y_train_predicted = knn_model.predict(X_train)
# print "Error Rate for kNN (Cropped): ", ml_aux.get_error_rate(y_train, y_train_predicted)

y_validation_predicted =  knn_model.predict(X_validation)
print "Error Rate for kNN on Validation (t1): ", ml_aux.get_error_rate(y_validation, y_validation_predicted)


# Start Naive Bayes Classification
print "Performing Naive Bayes Classification:"
from sklearn.naive_bayes import GaussianNB
naivebayes_model = GaussianNB()
naivebayes_model.fit(X_cropped, y_cropped)
y_validation_predicted = naivebayes_model.predict(X_validation)
print "Naive Bayes Error Rate on Validation (t1): ", ml_aux.get_error_rate(y_validation, y_validation_predicted)


# Start SVM Classification
print "Performing SVM Classification:"
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf' ,probability=True, max_iter=100000)
svm_model.fit(X_cropped, y_cropped)
y_train_predicted = svm_model.predict(X_train)
print "SVM Error rate on training data (t1): ", ml_aux.get_error_rate(y_train, y_train_predicted)
# ml_aux.plot_confusion_matrix(y_train, y_train_predicted, "CM SVM Training (t1)")
# plt.show()

y_validation_predicted = svm_model.predict(X_validation)
print "SVM Error rate on validation (t1): ", ml_aux.get_error_rate(y_validation, y_validation_predicted)


# Start k nearest Centroid Classification
print "Performing kNC Classification:"
from sklearn.neighbors.nearest_centroid import NearestCentroid
knnc_model = NearestCentroid()
knnc_model.fit(X_cropped, y_cropped)
y_validation_predicted = knnc_model.predict(X_validation)
print "Error Rate on kNNC (t1) Validation:  ", ml_aux.get_error_rate(y_validation, y_validation_predicted)

# Start Bagging Classification
print "Performing Bagging Classification:"
# Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Bagging
bagging1 = BaggingClassifier(KNeighborsClassifier(n_neighbors=2),max_samples=1.0, max_features=0.1)
bagging1.fit(X_cropped, y_cropped)
y_validation_predicted = bagging1.predict(X_validation)
print "Error Rate kNN with Baggging Validation: ", ml_aux.get_error_rate(y_validation, y_validation_predicted)

