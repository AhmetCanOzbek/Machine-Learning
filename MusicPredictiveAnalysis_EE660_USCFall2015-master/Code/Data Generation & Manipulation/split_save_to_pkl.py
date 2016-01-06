__author__ = "Can Ozbek"

import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import ml_aux_functions as ml_aux

"""
"msd.pkl": 1 million song dataset with songhotnesss dropped
"""

header_names = [
    "Genre", "Track ID",
    'AvgBarDuration','Loudness', 'Tempo','ArtistFamiliarity','ArtistHotttnesss','SongHotttnesss',
    'Mode[0]','Mode[1]','Year',
    #Key features
    'Key[0]','Key[1]','Key[2]','Key[3]','Key[4]','Key[5]',
    'Key[6]','Key[7]','Key[8]','Key[9]','Key[10]','Key[11]',
    #Picthes Mean
    'PicthesMean[0]','PicthesMean[1]','PicthesMean[2]','PicthesMean[3]','PicthesMean[4]','PicthesMean[5]',
    'PicthesMean[6]','PicthesMean[7]','PicthesMean[8]','PicthesMean[9]','PicthesMean[10]','PicthesMean[11]',
    #Pitches Variance
    'PitchesVar[0]','PitchesVar[1]','PitchesVar[2]','PitchesVar[3]','PitchesVar[4]','PitchesVar[5]',
    'PitchesVar[6]','PitchesVar[7]','PitchesVar[8]','PitchesVar[9]','PitchesVar[10]','PitchesVar[11]',
    #Timbre Mean
    'TimbreMean[0]','TimbreMean[1]','TimbreMean[2]','TimbreMean[3]','TimbreMean[4]','TimbreMean[5]',
    'TimbreMean[6]','TimbreMean[7]','TimbreMean[8]','TimbreMean[9]','TimbreMean[10]','TimbreMean[11]',
    #Timbre Variance
    'TimbreVar[0]','TimbreVar[1]','TimbreVar[2]','TimbreVar[3]','TimbreVar[4]','TimbreVar[5]',
    'TimbreVar[6]','TimbreVar[7]','TimbreVar[8]','TimbreVar[9]','TimbreVar[10]','TimbreVar[11]',
    #Time Signature
    'TimeSig[0]', 'TimeSig[1]', 'TimeSig[2]', 'TimeSig[3]', 'TimeSig[4]', 'TimeSig[5]']

#Read the 1million song dataset
df = pd.read_csv("/Users/ahmetcanozbek/Desktop/660Stuff/MSD_Final_Dataset_For_EE660_Final.bin",
                 header = None, delimiter = "|", names = header_names)

#Drop the songhotnesss column
df = df.drop("SongHotttnesss", axis = 1)

#Save the general 1 million song data set frame with songhotttness dropped
df.to_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd.pkl")
print "Saving 'msd.pkl' done."

#Save the 20% of data as Test Data
from sklearn.cross_validation import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_test.to_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_test.pkl")
print "Saving 'msd_test.pkl' done.", df_test.shape

#*Training data 80%
#Save the training data as a whole
df_train.to_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train.pkl")
print "Saving 'msd_train.pkl' done.", df_train.shape

#Split the training data into 5 parts and save
#1)1922 - 1963 (t1)
df_train_t1 = df_train[ (df_train["Year"]>=1922) & (df_train["Year"]<=1963) ]
df_train_t1.to_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t1.pkl")
print "Saving 'msd_train_t1.pkl' done.", df_train_t1.shape

#2)1964 - 1980 (t2)
df_train_t2 = df_train[ (df_train["Year"]>=1964) & (df_train["Year"]<=1980) ]
df_train_t2.to_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t2.pkl")
print "Saving 'msd_train_t2.pkl' done.", df_train_t2.shape

#3)1981 - 1991 (t3)
df_train_t3 = df_train[ (df_train["Year"]>=1981) & (df_train["Year"]<=1991) ]
df_train_t3.to_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t3.pkl")
print "Saving 'msd_train_t3.pkl' done.", df_train_t3.shape

#4)1992 - 2002 (t4)
df_train_t4 = df_train[ (df_train["Year"]>=1992) & (df_train["Year"]<=2002) ]
df_train_t4.to_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t4.pkl")
print "Saving 'msd_train_t4.pkl' done.", df_train_t4.shape

#5)2003 - 2011 (t5)
df_train_t5 = df_train[ (df_train["Year"]>=2003) & (df_train["Year"]<=2011) ]
df_train_t5.to_pickle("/Users/ahmetcanozbek/Desktop/660Stuff/msd_train_t5.pkl")
print "Saving 'msd_train_t5.pkl' done.", df_train_t5.shape


print "Sum: ", df_train_t1.shape[0]+df_train_t2.shape[0]+df_train_t3.shape[0]+df_train_t4.shape[0]+df_train_t5.shape[0]






