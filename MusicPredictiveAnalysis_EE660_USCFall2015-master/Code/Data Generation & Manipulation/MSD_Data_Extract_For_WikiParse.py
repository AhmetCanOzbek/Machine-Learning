# !/usr/bin/env python

__author__ = 'NishantNath'

# Note : This version does not have latest info. Refer to part files like MSD_Data_Extract_Wiki_a2f.py, etc

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
import unicodedata



if __name__ == '__main__':

    print('----- started -----')

    # setting directory where file is located as working directory
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)

    # global variables declared here
    #

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

                #data extract 1
                track_id_str = hdf5_getters.get_track_id(h5file)
                song_id_str = hdf5_getters.get_song_id(h5file)
                album_name_str = unicodedata.normalize('NFKD', unicode(hdf5_getters.get_release(h5file),encoding='ASCII',errors='ignore'))
                album_name_str = str(album_name_str).replace(","," ").replace("'","").replace("-"," ").replace("("," ").replace(")"," ").replace("/"," ").replace("\\"," ")
                year_str = str(hdf5_getters.get_year(h5file))

                counter += 1
                file_io_counter += 1
                if file_io_counter % 100 == 0:
                    if counter == 1:
                        my_array_data_extract_1 = numpy.array([track_id_str, song_id_str, album_name_str, year_str])
                    else :
                        my_array_data_extract_1 = numpy.vstack((my_array_data_extract_1,numpy.array([track_id_str, song_id_str, album_name_str, year_str])))

                    f_handle = file('msd_data_extract_1.bin','a')
                    numpy.savetxt(f_handle, my_array_data_extract_1, delimiter='|',fmt='%s')
                    f_handle.close()

                    del my_array_data_extract_1
                    counter = 0
                    file_io_counter = 0
                else:
                    if counter == 1:
                        my_array_data_extract_1 = numpy.array([track_id_str, song_id_str, album_name_str, year_str])
                    else :
                        my_array_data_extract_1 = numpy.vstack((my_array_data_extract_1,numpy.array([track_id_str, song_id_str, album_name_str, year_str])))

                h5file.close()

            else: # redundant but left as placeholder
                pass

    elapsed_time = time.time() - start_time
    print("elapsed time : ",elapsed_time)

    # print remaining data to file
    try:
        my_array_data_extract_1
    except NameError:
        print "my_array_data_extract_1 does not exist currently"
    else:
        f_handle = file('msd_data_extract_1.bin','a')
        numpy.savetxt(f_handle, my_array_data_extract_1, delimiter='|',fmt='%s')
        f_handle.close()

    print('----- done -----')