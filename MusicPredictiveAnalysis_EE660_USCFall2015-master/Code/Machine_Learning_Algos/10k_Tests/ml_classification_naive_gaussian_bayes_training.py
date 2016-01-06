# !/usr/bin/env python

__author__ = 'NishantNath'


'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py, find_second_max_value.py
Required packages : numpy, pandas, matplotlib, sklearn, pickle

Steps:
1.

# Uses naive bayes gaussian method for classification training
'''

import pandas
import matplotlib.pyplot as mpyplot
import pylab
import numpy
import sklearn
import pickle

import crop_rock

# [0: 'CLASSICAL', 1: 'METAL', 2: 'DANCE', 3: 'JAZZ']
# [4:'FOLK', 5: 'SOUL', 6: 'ROCK', 7: 'POP', 8: 'BLUES']



if __name__ == '__main__':
    print '--- started ---'

    input1 = pickle.load(open("msd_train_t1.pkl", "rb"))
    input2 = pickle.load(open("msd_train_t2.pkl", "rb"))
    input3 = pickle.load(open("msd_train_t3.pkl", "rb"))
    input4 = pickle.load(open("msd_train_t4.pkl", "rb"))
    input5 = pickle.load(open("msd_train_t5.pkl", "rb"))
    # print input1.shape[0]
    # input = pickle.load(open("msd_train.pkl", "rb"))

    maxval1 = crop_rock.find_second_max_value(input1)
    maxval2 = crop_rock.find_second_max_value(input2)
    maxval3 = crop_rock.find_second_max_value(input3)
    maxval4 = crop_rock.find_second_max_value(input4)
    maxval5 = crop_rock.find_second_max_value(input5)
    # print maxval1
    # maxval = crop_rock.find_second_max_value(input)

    filtered1 = crop_rock.drop_excess_rows(input1, maxval1)
    filtered2 = crop_rock.drop_excess_rows(input2, maxval2)
    filtered3 = crop_rock.drop_excess_rows(input3, maxval3)
    filtered4 = crop_rock.drop_excess_rows(input4, maxval4)
    filtered5 = crop_rock.drop_excess_rows(input5, maxval5)
    # print filtered1.shape[0]
    # filtered = crop_rock.drop_excess_rows(input, maxval)

    #handling missing data
    filtered1 = filtered1[filtered1['Genre']!='UNCAT']; filtered1 = filtered1.dropna()
    filtered2 = filtered2[filtered2['Genre']!='UNCAT']; filtered2 = filtered2.dropna()
    filtered3 = filtered3[filtered3['Genre']!='UNCAT']; filtered3 = filtered3.dropna()
    filtered4 = filtered4[filtered4['Genre']!='UNCAT']; filtered4 = filtered4.dropna()
    filtered5 = filtered5[filtered5['Genre']!='UNCAT']; filtered5 = filtered5.dropna()
    # print filtered1
    # filtered = filtered[filtered['Genre']!='UNCAT']; filtered.dropna()

    # range(2,76) means its goes from col 2 to col 75
    df_input1_data = filtered1[list(range(2,76))].as_matrix()
    df_input1_target = filtered1[list(range(0,1))].as_matrix()

    df_input2_data = filtered2[list(range(2,76))].as_matrix()
    df_input2_target = filtered2[list(range(0,1))].as_matrix()

    df_input3_data = filtered3[list(range(2,76))].as_matrix()
    df_input3_target = filtered3[list(range(0,1))].as_matrix()

    df_input4_data = filtered4[list(range(2,76))].as_matrix()
    df_input4_target = filtered4[list(range(0,1))].as_matrix()

    df_input5_data = filtered5[list(range(2,76))].as_matrix()
    df_input5_target = filtered5[list(range(0,1))].as_matrix()

    # df_input_data = filtered[list(range(2,76))].as_matrix()
    # df_input_target = filtered[list(range(0,1))].as_matrix()

    # Naive Gaussian Bayes
    from sklearn.naive_bayes import GaussianNB

    # Naive Bayes (gaussian model)
    gnb1 = GaussianNB()
    gnb1.fit(df_input1_data,numpy.ravel(df_input1_target))
    pickle.dump(gnb1, open('model_naiveGaussianBayes_t1.pkl', 'wb'))

    gnb2 = GaussianNB()
    gnb2.fit(df_input2_data,numpy.ravel(df_input2_target))
    pickle.dump(gnb2, open('model_naiveGaussianBayes_t2.pkl', 'wb'))

    gnb3 = GaussianNB()
    gnb3.fit(df_input3_data,numpy.ravel(df_input3_target))
    pickle.dump(gnb3, open('model_naiveGaussianBayes_t3.pkl', 'wb'))

    gnb4 = GaussianNB()
    gnb4.fit(df_input4_data,numpy.ravel(df_input4_target))
    pickle.dump(gnb4, open('model_naiveGaussianBayes_t4.pkl', 'wb'))

    gnb5 = GaussianNB()
    gnb5.fit(df_input5_data,numpy.ravel(df_input5_target))
    pickle.dump(gnb5, open('model_naiveGaussianBayes_t5.pkl', 'wb'))

    # gnb = GaussianNB()
    # gnb.fit(df_input_data,numpy.ravel(df_input_target))
    # pickle.dump(gnb, open('model_naiveGaussianBayes_train.pkl', 'wb'))

    predicted1 = gnb1.predict(df_input1_data)
    predicted2 = gnb2.predict(df_input2_data)
    predicted3 = gnb3.predict(df_input3_data)
    predicted4 = gnb4.predict(df_input4_data)
    predicted5 = gnb5.predict(df_input5_data)
    # predicted = gnb.predict(df_input_data)

    matches1 = (predicted1 == [item for sublist in df_input1_target for item in sublist])
    matches2 = (predicted2 == [item for sublist in df_input2_target for item in sublist])
    matches3 = (predicted3 == [item for sublist in df_input3_target for item in sublist])
    matches4 = (predicted4 == [item for sublist in df_input4_target for item in sublist])
    matches5 = (predicted5 == [item for sublist in df_input5_target for item in sublist])
    # matches = (predicted == [item for sublist in df_input_target for item in sublist])

    print 'using excess rock & uncats removed'
    print "Accuracy of T1 : ", (matches1.sum() / float(len(matches1)))
    print "Accuracy of T2 : ", (matches2.sum() / float(len(matches2)))
    print "Accuracy of T3 : ", (matches3.sum() / float(len(matches3)))
    print "Accuracy of T4 : ", (matches4.sum() / float(len(matches4)))
    print "Accuracy of T5 : ", (matches5.sum() / float(len(matches5)))
    # print "Accuracy of Training : ", (matches.sum() / float(len(matches)))

    x1 = input1[input1['Genre']!='UNCAT'].dropna()
    x_df_input1_data = x1[list(range(2,76))].as_matrix()
    x_df_input1_target = x1[list(range(0,1))].as_matrix()

    x2 = input2[input2['Genre']!='UNCAT'].dropna()
    x_df_input2_data = x2[list(range(2,76))].as_matrix()
    x_df_input2_target = x2[list(range(0,1))].as_matrix()

    x3 = input3[input3['Genre']!='UNCAT'].dropna()
    x_df_input3_data = x3[list(range(2,76))].as_matrix()
    x_df_input3_target = x3[list(range(0,1))].as_matrix()

    x4 = input4[input4['Genre']!='UNCAT'].dropna()
    x_df_input4_data = x4[list(range(2,76))].as_matrix()
    x_df_input4_target = x4[list(range(0,1))].as_matrix()

    x5 = input5[input5['Genre']!='UNCAT'].dropna()
    x_df_input5_data = x5[list(range(2,76))].as_matrix()
    x_df_input5_target = x5[list(range(0,1))].as_matrix()

    # x = input[input['Genre']!='UNCAT'].dropna()
    # x_df_input_data = x[list(range(2,76))].as_matrix()
    # x_df_input_target = x[list(range(0,1))].as_matrix()

    predicted1 = gnb1.predict(x_df_input1_data)
    predicted2 = gnb2.predict(x_df_input2_data)
    predicted3 = gnb3.predict(x_df_input3_data)
    predicted4 = gnb4.predict(x_df_input4_data)
    predicted5 = gnb5.predict(x_df_input5_data)
    # predicted = gnb.predict(x_df_input_data)

    matches1 = (predicted1 == [item for sublist in x_df_input1_target for item in sublist])
    matches2 = (predicted2 == [item for sublist in x_df_input2_target for item in sublist])
    matches3 = (predicted3 == [item for sublist in x_df_input3_target for item in sublist])
    matches4 = (predicted4 == [item for sublist in x_df_input4_target for item in sublist])
    matches5 = (predicted5 == [item for sublist in x_df_input5_target for item in sublist])
    # matches = (predicted == [item for sublist in x_df_input_target for item in sublist])

    print 'using uncats removed'
    print "Accuracy of T1 : ", (matches1.sum() / float(len(matches1)))
    print "Accuracy of T2 : ", (matches2.sum() / float(len(matches2)))
    print "Accuracy of T3 : ", (matches3.sum() / float(len(matches3)))
    print "Accuracy of T4 : ", (matches4.sum() / float(len(matches4)))
    print "Accuracy of T5 : ", (matches5.sum() / float(len(matches5)))
    # print "Accuracy of Training : ", (matches.sum() / float(len(matches)))

print '--- done ---'