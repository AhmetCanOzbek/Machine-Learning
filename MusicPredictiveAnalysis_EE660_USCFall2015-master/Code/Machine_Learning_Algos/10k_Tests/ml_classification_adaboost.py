__author__ = "Can Ozbek"

import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import ml_aux_functions as ml_aux



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


#Start Adaboost Classification
print "*AdaBoost Classification"
#Adaboost model
from sklearn.ensemble import AdaBoostClassifier
adaboost_model = AdaBoostClassifier(n_estimators=100)
#Make the data ready
y = df_merged["Genre"]
X = df_merged.drop(["Genre","Song ID","Track ID"], axis = 1)
#Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
#Train
adaboost_model.fit(X_train,y_train)
#Predict
y_train_predicted = adaboost_model.predict(X_train)
y_test_predicted = adaboost_model.predict(X_test)

print "Number of Train Samples: ", (y_train.shape[0])
print "Number of Test Samples: ", (y_test.shape[0])

print "Train Classification Rate: ", (sum(y_train_predicted == y_train)) / float(y_train.shape[0])
print "Test Classification Rate: ", (sum(y_test_predicted == y_test)) / float(y_test.shape[0])

print ml_aux.getUniqueCount(y_train)
print ml_aux.getUniqueCount(y_test)

print "try func: ", ml_aux.get_error_rate(y_train, y_train_predicted)

print ml_aux.plot_confusion_matrix(y_train,y_train_predicted,"Train")
plt.show()


ml_aux.plot_confusion_matrix(y_test,y_test_predicted,"Test")
plt.show()



