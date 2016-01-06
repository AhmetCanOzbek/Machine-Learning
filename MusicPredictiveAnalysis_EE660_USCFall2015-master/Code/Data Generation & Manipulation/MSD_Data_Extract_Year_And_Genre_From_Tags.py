#!/usr/bin/env python

__author__ = 'NishantNath'

'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py
Required packages : os, numpy, time

1.

# This script extracts the various combinations of data from MSD (whole set) and saves as different comma separated bin files

'''

import os
import hdf5_getters
import numpy
import time
import re
from collections import Counter
import operator


if __name__ == '__main__':

    print('----- started -----')

    # setting directory where file is located as working directory
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)

    # global variables declared here
    my_genre = ['CLASSICAL','METAL','HIPHOP','DANCE','JAZZ','FOLK','SOUL','ROCK','POP','BLUES']

    start_time=time.time()

    process_completion = 0
    counter = 0
    for root, dirs, files in os.walk(os.getcwd()):
        for name in files:
            if name.endswith(".h5"):
                if process_completion % 400 == 0:
                    print "done :", process_completion/400.0, "%"
                    process_completion += 1
                else:
                    process_completion += 1

                tempPath = os.path.abspath(os.path.join(root,name))
                h5file = hdf5_getters.open_h5_file_read(tempPath)

                #meta info
                track_id_str = hdf5_getters.get_track_id(h5file)
                song_id_str = hdf5_getters.get_song_id(h5file)
                year = hdf5_getters.get_year(h5file)

                #genre info
                artist_terms = hdf5_getters.get_artist_terms(h5file)
                artist_mbtags = hdf5_getters.get_artist_mbtags(h5file)
                artist_mbtags_count = hdf5_getters.get_artist_mbtags_count(h5file)
                artist_terms_freq = hdf5_getters.get_artist_terms_freq(h5file)
                artist_terms_weight = hdf5_getters.get_artist_terms_weight(h5file)

                #factoring in mbtags counts & thresholding artist_terms
                idx = 0
                for countVal in artist_mbtags_count:
                    for rep in range(0,countVal-1):
                        artist_mbtags = numpy.append(artist_mbtags,artist_mbtags[idx])
                    idx += 1

                for i in range(0,len(artist_terms)):
                    if artist_terms_freq[i]*artist_terms_weight[i] < 0.16: #0.4*0.4 = 0.16
                        artist_terms[i]=""

                # finding genre
                tags = " ".join(artist_mbtags)+" "+ " ".join(artist_terms)
                tags = tags.replace("   "," ").replace("  "," ").replace(" & ","&")
                tags = tags.replace("HIP HOP","HIPHOP").replace("HIP-HOP","HIPHOP").replace("HIP/HOP","HIPHOP").replace("HIP\HOP","HIPHOP").replace("HIP.HOP","HIPHOP").replace("HIP+HOP","HIPHOP").replace("HIP&HOP","HIPHOP")
                tags = tags.replace("BLUE","BLUES").replace("BLUESS","BLUES")
                tags = tags.replace(","," ").replace("'"," ").replace("`"," ").replace("-"," ").replace(":"," ").replace("."," ").replace(";"," ").replace("+"," ")
                words = re.findall(r'\w+', tags)
                if len(words) != 0:
                    cap_words = [word.upper() for word in words]
                    word_counts = Counter(cap_words)
                    genre = max(word_counts.iteritems(), key=operator.itemgetter(1))[0]

                    if genre not in my_genre:
                        genre = 'UNCAT'
                else:
                    genre='UNCAT'

                h5file.close()

                counter += 1
                if counter == 1:
                    my_array = numpy.array([track_id_str, song_id_str, genre, year])
                else :
                    my_array = numpy.vstack((my_array,numpy.array([track_id_str, song_id_str, genre, year])))

            else: # redundant but left as placeholder
                pass

    elapsed_time = time.time() - start_time
    print("elapsed time : ",elapsed_time)

    numpy.savetxt('msd_data_extract_2.bin', my_array, delimiter='|',fmt='%s')
    print('----- done -----')