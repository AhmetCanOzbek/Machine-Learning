# !/usr/bin/env python

__author__ = 'NishantNath'


'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py, find_second_max_value.py
Required packages : numpy, pandas, matplotlib, sklearn, pickle

Steps:
1.

# Uses k-means method for classification training
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
    from sklearn.mixture import GMM

    # Simple PCA
    from sklearn.decomposition import PCA
    pca1 = PCA(n_components=6) #from optimal pca components chart n_components=6
    pca2 = PCA(n_components=6) #from optimal pca components chart n_components=6
    pca3 = PCA(n_components=6) #from optimal pca components chart n_components=6
    pca4 = PCA(n_components=6) #from optimal pca components chart n_components=6
    pca5 = PCA(n_components=6) #from optimal pca components chart n_components=6
    # pca = PCA(n_components=6) #from optimal pca components chart n_components=6
    pca1.fit(df_input1_data)
    pca2.fit(df_input2_data)
    pca3.fit(df_input3_data)
    pca4.fit(df_input4_data)
    pca5.fit(df_input5_data)
    # pca.fit(df_input_data)

    # Reduced Feature Set
    df_input1_data = pca1.transform(df_input1_data)
    df_input2_data = pca2.transform(df_input2_data)
    df_input3_data = pca3.transform(df_input3_data)
    df_input4_data = pca4.transform(df_input4_data)
    df_input5_data = pca5.transform(df_input5_data)
    # df_input_data = pca.transform(df_input_data)


    # Naive Bayes (gaussian model)
    gmm1 = gmm = GMM(n_components=9, covariance_type='tied')
    gmm1.fit(df_input1_data,numpy.ravel(df_input1_target))
    pickle.dump(gmm1, open('model_gmm_t1.pkl', 'wb'))

    gmm2 = gmm = GMM(n_components=9, covariance_type='tied')
    gmm2.fit(df_input2_data,numpy.ravel(df_input2_target))
    pickle.dump(gmm2, open('model_gmm_t2.pkl', 'wb'))

    gmm3 = gmm = GMM(n_components=9, covariance_type='tied')
    gmm3.fit(df_input3_data,numpy.ravel(df_input3_target))
    pickle.dump(gmm3, open('model_gmm_t3.pkl', 'wb'))

    gmm4 = gmm = GMM(n_components=9, covariance_type='tied')
    gmm4.fit(df_input4_data,numpy.ravel(df_input4_target))
    pickle.dump(gmm4, open('model_gmm_t4.pkl', 'wb'))

    gmm5 = gmm = GMM(n_components=9, covariance_type='tied')
    gmm5.fit(df_input5_data,numpy.ravel(df_input5_target))
    pickle.dump(gmm5, open('model_gmm_t5.pkl', 'wb'))

    # gmm = KMeans(n_clusters=5, random_state=RandomState(9)
    # gmm.fit(df_input_data,numpy.ravel(df_input_target))
    # pickle.dump(gmm, open('model_gmm_train.pkl', 'wb'))

    predicted1 = gmm1.predict(df_input1_data)
    predicted2 = gmm2.predict(df_input2_data)
    predicted3 = gmm3.predict(df_input3_data)
    predicted4 = gmm4.predict(df_input4_data)
    predicted5 = gmm5.predict(df_input5_data)
    # predicted = gmm.predict(df_input_data)

    matches1 = (predicted1 == [item for sublist in df_input1_target for item in sublist])
    matches2 = (predicted2 == [item for sublist in df_input2_target for item in sublist])
    matches3 = (predicted3 == [item for sublist in df_input3_target for item in sublist])
    matches4 = (predicted4 == [item for sublist in df_input4_target for item in sublist])
    matches5 = (predicted5 == [item for sublist in df_input5_target for item in sublist])
    # matches = (predicted == [item for sublist in df_input_target for item in sublist])

    print 'using excess rock & uncats removed'

    print predicted1, type(predicted1)
    print matches1, type(matches1)

    # print "Accuracy of T1 : ", (matches1.count() / float(len(matches1)))
    # print "Accuracy of T2 : ", (matches2.count() / float(len(matches2)))
    # print "Accuracy of T3 : ", (matches3.count() / float(len(matches3)))
    # print "Accuracy of T4 : ", (matches4.count() / float(len(matches4)))
    # print "Accuracy of T5 : ", (matches5.count() / float(len(matches5)))
    # # print "Accuracy of Training : ", (matches.count() / float(len(matches)))
    #
    # x1 = input1[input1['Genre']!='UNCAT'].dropna()
    # x_df_input1_data = x1[list(range(2,76))].as_matrix()
    # x_df_input1_target = x1[list(range(0,1))].as_matrix()
    #
    # x2 = input2[input2['Genre']!='UNCAT'].dropna()
    # x_df_input2_data = x2[list(range(2,76))].as_matrix()
    # x_df_input2_target = x2[list(range(0,1))].as_matrix()
    #
    # x3 = input3[input3['Genre']!='UNCAT'].dropna()
    # x_df_input3_data = x3[list(range(2,76))].as_matrix()
    # x_df_input3_target = x3[list(range(0,1))].as_matrix()
    #
    # x4 = input4[input4['Genre']!='UNCAT'].dropna()
    # x_df_input4_data = x4[list(range(2,76))].as_matrix()
    # x_df_input4_target = x4[list(range(0,1))].as_matrix()
    #
    # x5 = input5[input5['Genre']!='UNCAT'].dropna()
    # x_df_input5_data = x5[list(range(2,76))].as_matrix()
    # x_df_input5_target = x5[list(range(0,1))].as_matrix()
    #
    # # x = input[input['Genre']!='UNCAT'].dropna()
    # # x_df_input_data = x[list(range(2,76))].as_matrix()
    # # x_df_input_target = x[list(range(0,1))].as_matrix()
    #
    # predicted1 = gmm1.predict(x_df_input1_data)
    # predicted2 = gmm2.predict(x_df_input2_data)
    # predicted3 = gmm3.predict(x_df_input3_data)
    # predicted4 = gmm4.predict(x_df_input4_data)
    # predicted5 = gmm5.predict(x_df_input5_data)
    # # predicted = gmm.predict(x_df_input_data)
    #
    # matches1 = (predicted1 == [item for sublist in x_df_input1_target for item in sublist])
    # matches2 = (predicted2 == [item for sublist in x_df_input2_target for item in sublist])
    # matches3 = (predicted3 == [item for sublist in x_df_input3_target for item in sublist])
    # matches4 = (predicted4 == [item for sublist in x_df_input4_target for item in sublist])
    # matches5 = (predicted5 == [item for sublist in x_df_input5_target for item in sublist])
    # # matches = (predicted == [item for sublist in x_df_input_target for item in sublist])
    #
    # print 'using uncats removed'
    # print "Accuracy of T1 : ", (matches1.count() / float(len(matches1)))
    # print "Accuracy of T2 : ", (matches2.count() / float(len(matches2)))
    # print "Accuracy of T3 : ", (matches3.count() / float(len(matches3)))
    # print "Accuracy of T4 : ", (matches4.count() / float(len(matches4)))
    # print "Accuracy of T5 : ", (matches5.count() / float(len(matches5)))
    # # print "Accuracy of Training : ", (matches.count() / float(len(matches)))

print '--- done ---'