__author__ = "Can Ozbek"

import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import ml_aux_functions as ml_aux

#Read the files
df = pd.read_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd.pkl")
df_train = pd.read_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train.pkl") # 80%
df_test = pd.read_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_test.pkl") # 20%

df_train_t1 = pd.read_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t1.pkl")
df_train_t2 = pd.read_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t2.pkl")
df_train_t3 = pd.read_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t3.pkl")
df_train_t4 = pd.read_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t4.pkl")
df_train_t5 = pd.read_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t5.pkl")
print "Reading Done."

print "Histogram: "
print ml_aux.getUniqueCount(df_train_t1["Genre"])

ml_aux.plot_histogram(ml_aux.getUniqueCount(df_train_t1["Genre"]))
plt.show()

