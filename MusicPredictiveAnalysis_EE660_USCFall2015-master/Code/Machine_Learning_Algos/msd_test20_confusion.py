#Script for final testing
__author__ = "Can Ozbek"

import pandas as pd
import pickle
import numpy as np
import pylab
import matplotlib.pyplot as plt
import ml_aux_functions as ml_aux
from sklearn.metrics import confusion_matrix

#Read the test file
# df_test = pd.read_pickle("msd_test.pkl','rb'))   # 20%
df_test = pickle.load(open('msd_test.pkl','rb'))   # 20%
print "DEBUG: file read."

#Get rid of the rows that have missing values (nan) and UNCAT
df_test = df_test[ df_test["Genre"] != "UNCAT" ]
df_test = df_test.dropna()
print "UNCAT and nan cleaning done."


#Split the testing data into 6 parts and save
#1)1922 - 1963 (t1)
df_test_t1 = df_test[ (df_test["Year"]>=1922) & (df_test["Year"]<=1963) ]
y_test_t1 = df_test_t1["Genre"]
X_test_t1 = df_test_t1.drop(["Genre", "Track ID", "Year"], axis=1)
#2)1964 - 1980 (t2)
df_test_t2 = df_test[ (df_test["Year"]>=1964) & (df_test["Year"]<=1980) ]
y_test_t2 = df_test_t2["Genre"]
X_test_t2 = df_test_t2.drop(["Genre", "Track ID", "Year"], axis=1)
#3)1981 - 1991 (t3)
df_test_t3 = df_test[ (df_test["Year"]>=1981) & (df_test["Year"]<=1991) ]
y_test_t3 = df_test_t3["Genre"]
X_test_t3 = df_test_t3.drop(["Genre", "Track ID", "Year"], axis=1)
#4)1992 - 2002 (t4)
df_test_t4 = df_test[ (df_test["Year"]>=1992) & (df_test["Year"]<=2002) ]
y_test_t4 = df_test_t4["Genre"]
X_test_t4 = df_test_t4.drop(["Genre", "Track ID", "Year"], axis=1)
#5)2003 - 2011 (t5)
df_test_t5 = df_test[ (df_test["Year"]>=2003) & (df_test["Year"]<=2011) ]
y_test_t5 = df_test_t5["Genre"]
X_test_t5 = df_test_t5.drop(["Genre", "Track ID", "Year"], axis=1)
#6)No Time Info
df_test_not = df_test[ df_test["Year"] < 1920 ]
y_test_not = df_test_not["Genre"]
X_test_not = df_test_not.drop(["Genre", "Track ID", "Year"], axis=1)
print "Splitting into 6 parts done."


#Read the model .pkl files
model_t1 = pickle.load(open('t1_random_forest.pkl','rb'))
model_t2 = pickle.load(open('t2_svm.pkl','rb'))
model_t3 = pickle.load(open('t3_svm.pkl','rb'))
model_t4 = pickle.load(open('t4_svm.pkl','rb'))
model_t5 = pickle.load(open('t5_random_forest.pkl','rb'))
model_not = pickle.load(open('fullset_forest.pkl','rb'))
print "Reading the model '.pkl' files done."

#Classification, get prediction
y_test_t1_predicted = model_t1.predict(X_test_t1)
y_test_t2_predicted = model_t2.predict(X_test_t2)
y_test_t3_predicted = model_t3.predict(X_test_t3)
y_test_t4_predicted = model_t4.predict(X_test_t4)
y_test_t5_predicted = model_t5.predict(X_test_t5)
y_test_not_predicted = model_not.predict(X_test_not)
print "Predictions with models done."


totalNumberOfErrors =   sum(y_test_t1 != y_test_t1_predicted) + \
                        sum(y_test_t2 != y_test_t2_predicted) + \
                        sum(y_test_t3 != y_test_t3_predicted) + \
                        sum(y_test_t4 != y_test_t4_predicted) + \
                        sum(y_test_t5 != y_test_t5_predicted) + \
                        sum(y_test_not != y_test_not_predicted)


totalNumberOfSamples = df_test_t1.shape[0]+df_test_t2.shape[0]+df_test_t3.shape[0]+df_test_t4.shape[0]+df_test_t5.shape[0] + df_test_not.shape[0]
error_rate = totalNumberOfErrors / float(totalNumberOfSamples)

print "Total Number of Samples: ", totalNumberOfSamples
print "Total Number of Errors: ", totalNumberOfErrors
print "Error Rate on Testing Set: ", error_rate

file = open("test20results.txt", "w")
file.write("Total Number of Samples: " + str(totalNumberOfSamples) + "\n")
file.write("Total Number of Errors: " + str(totalNumberOfErrors) + "\n")
file.write("Error Rate on Testing Set: "+ str(error_rate))
file.close()

merged_test = y_test_t1.append(y_test_t2)\
                       .append(y_test_t3)\
                       .append(y_test_t4)\
                       .append(y_test_t5)\
                       .append(y_test_not)


print "shapes"
print y_test_t1_predicted.shape
print y_test_t2_predicted.shape
print y_test_t3_predicted.shape
print y_test_t4_predicted.shape
print y_test_t5_predicted.shape
print y_test_not_predicted.shape



merged_test_predicted = np.hstack([y_test_t1_predicted,
                                  y_test_t2_predicted,
                                  y_test_t3_predicted,
                                  y_test_t4_predicted,
                                  y_test_t5_predicted,
                                  y_test_not_predicted])

#Plot confusion matrix
ml_aux.plot_confusion_matrix(merged_test,merged_test_predicted,"Confusion Matrix")
#Save the plot
plt.savefig("finaltest.png")














