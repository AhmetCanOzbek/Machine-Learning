__author__ = 'Arnav'

import numpy as np
import h5py
import combine_feat as combine
import matplotlib
import matplotlib.pyplot as plt

f_hiphop = h5py.File("G:\project660\Resources\MillionSongSubset\data\A\A\A\TRAAAAW128F429D538.h5", 'r')
f_classic = h5py.File("G:\project660\Resources\MillionSongSubset\data\B\H\H\TRBHHHC128F428428E.h5", 'r')
f_country = h5py.File("G:\project660\Resources\MillionSongSubset\data\B\C\E\TRBCEST128F9327FBF.h5", 'r')
f_metal = h5py.File("G:\project660\Resources\MillionSongSubset\data\B\C\R\TRBCRDW128F422D8BA.h5", 'r')
f_pop = h5py.File("G:\project660\Resources\MillionSongSubset\data\B\C\W\TRBCWNH128F93103EE.h5", 'r')
f = [f_hiphop, f_classic, f_country, f_metal, f_pop]
labs = ["Hip-hop", "Classical", "Country", "Metal", "Pop"]


"""
feat_pop = combine.combine_feature(np.array(f_pop['/analysis/segments_pitches']))
full_feat, axarr = plt.subplots(3, sharex=True, sharey=True)
axarr[0].set_title('Mean, Var and Std of 12 pitches')
#axarr.legend(loc='best')
axarr[0].plot(feat_pop[:12], label="Pop")
axarr[1].plot(feat_pop[12:24], label="Pop")
axarr[2].plot(feat_pop[24:], label="Pop")
full_feat.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in full_feat.axes[:-1]], visible=False)

#print(feat_pop)
plt.show()

"""

# Set up subplot figures
full_feat, axarr = plt.subplots(3, sharex=True)
#coloring the background to R/G/B for Mean/Var-and-Std plots
axarr[0].axvspan(0, 12, facecolor='r', alpha=0.2)
axarr[1].axvspan(0, 12, facecolor='g', alpha=0.2)
axarr[2].axvspan(0, 12, facecolor='b', alpha=0.2)
#adding annotation and axes labels to plots
axarr[0].annotate('Means', xy=(11, 0.1), xycoords='data', color='r')
axarr[1].annotate('Variances', xy=(11, 0.01), xycoords='data', color='g')
axarr[2].annotate('Standard Deviations', xy=(11, 0.04), xycoords='data', color='b')
axarr[0].set_title('Mean, Var and Std of 12 pitches for all segments')

# Combine features and plot them for each genre
counter=1   # this is how we initialize array for first iteration
for file, genre in zip(f, labs):
    if counter == 1:
        is_confident = np.array(file['/analysis/segments_confidence'])>0.5
        feat = combine.combine_feature(np.array(file['/analysis/segments_pitches']), is_confident)
        axarr[0].plot(feat[:12], label=genre)
        axarr[1].plot(feat[12:24], label=genre)
        axarr[2].plot(feat[24:], label=genre)
        axarr[0].legend(loc='upper right', fontsize='xx-small')
        counter += 1    # after feat is calculated for 1st file, we'll never enter this loop
    else:
        is_confident = np.array(file['/analysis/segments_confidence'])>0.5
        temp = combine.combine_feature(np.array(file['/analysis/segments_pitches']), is_confident)
        axarr[0].plot(temp[:12], label=genre)
        axarr[1].plot(temp[12:24], label=genre)
        axarr[2].plot(temp[24:], label=genre)
        axarr[0].legend(loc='upper right', fontsize='xx-small')

        feat = np.vstack((feat, temp))

# Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
full_feat.subplots_adjust(hspace=0.05)
#plt.setp([a.get_xticklabels() for a in full_feat.axes[:-1]], visible=False)
# Set Plot Labels and print figure
plt.xlabel('12 pitches value')
plt.ylabel('Normalized scores for each pitch', y=1.5)
plt.show()