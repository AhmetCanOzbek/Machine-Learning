#!/usr/bin/env python

__author__ = 'CanOzbek & NishantNath'

'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : hdf5_getters.py, featureExtractionFunctions.py
Required packages : os, numpy, time

1.

# This script extracts the various combinations of data from MSD (whole set) and saves as different comma separated bin files

'''

import os
import hdf5_getters
import time
import numpy
import featureExtractionFunctions


if __name__ == '__main__':

    print('----- started -----')

    # setting directory where file is located as working directory
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)

    feature_names = [
    "Track ID", "Song ID",
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

    # Put feature functions in a list in order
    featureFunctions = [featureExtractionFunctions.getBarDuration,
                        featureExtractionFunctions.getLoudness,
                        featureExtractionFunctions.getTempo,
                        featureExtractionFunctions.getArtistFamiliarity,
                        featureExtractionFunctions.getArtistHotttnesss,
                        featureExtractionFunctions.getSongHotttnesss,
                        featureExtractionFunctions.getMode,
                        featureExtractionFunctions.getYear,
                        featureExtractionFunctions.getKey,
                        featureExtractionFunctions.getSegmentPitchesMean,
                        featureExtractionFunctions.getSegmentPitchesVar,
                        featureExtractionFunctions.getSegmentTimbreMean,
                        featureExtractionFunctions.getSegmentTimbreVar,
                        featureExtractionFunctions.getTimeSignature]

    # Initialize Feature Matrix Data Structure with feature names as header, (going to be list of lists)
    featureMatrix = []
    print "feature_names length", len(feature_names)

    start_time=time.time()

    process_completion = 0
    counter = 0
    file_io_counter = 0
    header_counter = 0
    for root, dirs, files in os.walk(os.getcwd()):
        for name in files:
            if name.endswith(".h5"):
                if process_completion % 2350 == 0:
                    print "done :", process_completion/2350.0, "%"
                    process_completion += 1
                    header_counter += 1
                else:
                    process_completion += 1
                    header_counter += 1

                tempPath = os.path.abspath(os.path.join(root,name))
                h5SongFile = hdf5_getters.open_h5_file_read(tempPath)

                # create the sample row for feature matrix
                # initialize the list with track ID and song ID
                sampleRow = [hdf5_getters.get_track_id(h5SongFile),
                             hdf5_getters.get_song_id(h5SongFile)]

                cnt = 0
                for func in featureFunctions:
                    cnt += 1
                    # print "functionNumber: ", cnt
                    # print "BarStrctr: ", hdf5_getters.get_bars_start(h5SongFile)
                    sampleRow += func(h5SongFile)

                # Construct the feature matrix that contains all the songs...
                # Data structure for featureMatrix will be list of lists, so that...
                # Numpy will be able to convert it to a matrix in a way that...
                # All list elements will become rows
                # featureMatrix += [sampleRow]
                featureMatrix = [sampleRow]

                h5SongFile.close()

                file_io_counter += 1
                counter += 1
                if file_io_counter % 1000 == 0:
                    if counter == 1 and header_counter == 1:
                        # Make it numpy matrix
                        featureMatrixNumpy = numpy.array([feature_names] + featureMatrix)
                    elif counter == 1 and header_counter != 1:
                        # Make it numpy matrix
                        featureMatrixNumpy = numpy.array(featureMatrix)
                    else:
                        # Make it numpy matrix & stack it with previously generated matrix
                        featureMatrixNumpy = numpy.vstack((featureMatrixNumpy, numpy.array(featureMatrix)))

                    f_handle = file('msd_data_extraction_final.bin','a')
                    numpy.savetxt(f_handle, featureMatrixNumpy, delimiter='|',fmt='%s')
                    f_handle.close()

                    del featureMatrixNumpy
                    counter = 0
                    file_io_counter = 0
                else:
                    if counter == 1 and header_counter == 1:
                        # Make it numpy matrix
                        featureMatrixNumpy = numpy.array([feature_names] + featureMatrix)
                    elif counter == 1 and header_counter != 1:
                        # Make it numpy matrix
                        featureMatrixNumpy = numpy.array(featureMatrix)
                    else:
                        # Make it numpy matrix & stack it with previously generated matrix
                        featureMatrixNumpy = numpy.vstack((featureMatrixNumpy, numpy.array(featureMatrix)))

            else: # redundant but left as placeholder for non .h5 files
                pass

    elapsed_time = time.time() - start_time
    print("elapsed time : ",elapsed_time)

    print "Number of h5 files processed : ", process_completion

     # print remaining data to file
    try:
        featureMatrixNumpy
    except NameError:
        print "featureMatrixNumpy does not exist currently"
    else:
        f_handle = file('msd_data_extraction_final.bin','a')
        numpy.savetxt(f_handle, featureMatrixNumpy, delimiter='|',fmt='%s')
        f_handle.close()

    print('----- done -----')