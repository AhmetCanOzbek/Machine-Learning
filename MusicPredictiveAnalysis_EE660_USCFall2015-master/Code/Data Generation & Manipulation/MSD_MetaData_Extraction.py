# !/usr/bin/env python

__author__ = 'NishantNath'


'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py
Required packages : os, numpy, time

1. Meta Data
2. Song Level Data

# This script extracts the various combinations of metadata from MSD (whole set) and saves as different comma separated bin files

'''

import os
import hdf5_getters
import numpy
import time


if __name__ == '__main__':

    print('----- started -----')

    # setting directory where file is located as working directory
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)

    start_time=time.time()

    process_completion = 0
    counter = 0
    file_io_counter = 0
    for root, dirs, files in os.walk(os.getcwd()):
        for name in files:
            if name.endswith(".h5"):
                if process_completion % 10000 == 0:
                    print "done :", process_completion/10000.0, "%"
                    process_completion += 1
                else:
                    process_completion += 1

                tempPath = os.path.abspath(os.path.join(root,name))
                h5file = hdf5_getters.open_h5_file_read(tempPath)

                #meta info
                track_id_str = hdf5_getters.get_track_id(h5file).replace(",","")
                song_id_str = hdf5_getters.get_song_id(h5file).replace(",","")
                title_str = hdf5_getters.get_title(h5file).replace(","," ")
                artist_name_str = hdf5_getters.get_artist_name(h5file).replace(","," ")
                artist_location_str = hdf5_getters.get_artist_location(h5file).replace(","," ")

                # song-level data info
                duration = hdf5_getters.get_duration(h5file)
                key = hdf5_getters.get_key(h5file)
                mode = hdf5_getters.get_mode(h5file)
                tempo = hdf5_getters.get_tempo(h5file)
                time_signature = hdf5_getters.get_time_signature(h5file)

                h5file.close()

                counter += 1
                file_io_counter += 1
                if file_io_counter % 100 == 0:
                    if counter == 1:
                        my_array_metadata_extract_1 = numpy.array([track_id_str, song_id_str, title_str, artist_name_str, artist_location_str])
                        my_array_metadata_extract_2 = numpy.array([track_id_str, song_id_str, duration, key, mode, tempo, time_signature])
                    else :
                        my_array_metadata_extract_1 = numpy.vstack((my_array_metadata_extract_1,numpy.array([track_id_str, song_id_str, title_str, artist_name_str, artist_location_str])))
                        my_array_metadata_extract_2 = numpy.vstack((my_array_metadata_extract_2,numpy.array([track_id_str, song_id_str, duration, key, mode, tempo, time_signature])))

                    f_handle1 = file('msd_metadata_extract_1.bin','a')
                    f_handle2 = file('msd_metadata_extract_2.bin','a')
                    numpy.savetxt(f_handle1, my_array_metadata_extract_1, delimiter='|',fmt='%s')
                    numpy.savetxt(f_handle2, my_array_metadata_extract_2, delimiter='|',fmt='%s')
                    f_handle1.close()
                    f_handle2.close()

                    del my_array_metadata_extract_1
                    del my_array_metadata_extract_2
                    counter = 0
                    file_io_counter = 0
                else:
                    if counter == 1:
                        my_array_metadata_extract_1 = numpy.array([track_id_str, song_id_str, title_str, artist_name_str, artist_location_str])
                        my_array_metadata_extract_2 = numpy.array([track_id_str, song_id_str, duration, key, mode, tempo, time_signature])
                    else :
                        my_array_metadata_extract_1 = numpy.vstack((my_array_metadata_extract_1,numpy.array([track_id_str, song_id_str, title_str, artist_name_str, artist_location_str])))
                        my_array_metadata_extract_2 = numpy.vstack((my_array_metadata_extract_2,numpy.array([track_id_str, song_id_str, duration, key, mode, tempo, time_signature])))

                h5file.close()

            else: # redundant but left as placeholder
                pass

    elapsed_time = time.time() - start_time
    print("elapsed time : ", elapsed_time)

    # print remaining data to file1
    try:
        my_array_metadata_extract_1
    except NameError:
        print "my_array_metadata_extract_1 does not exist currently"
    else:
        f_handle1 = file('msd_metadata_extract_1.bin','a')
        numpy.savetxt(f_handle1, my_array_metadata_extract_1, delimiter='|',fmt='%s')
        f_handle1.close()

    # print remaining data to file2
    try:
        my_array_metadata_extract_2
    except NameError:
        print "my_array_metadata_extract_2 does not exist currently"
    else:
        f_handle1 = file('msd_metadata_extract_2.bin','a')
        numpy.savetxt(f_handle1, my_array_metadata_extract_2, delimiter='|',fmt='%s')
        f_handle2.close()

    print('----- done -----')