__author__ = "Can Ozbek"
import os
import hdf5_getters
import featureExtractionFunctions
import time
import numpy

abspath = os.path.abspath(os.getcwd())
dirname = os.path.dirname(abspath)
#cd into million song subset folder
dataFolderPath = "/Resources/MillionSongSubset/data"
os.chdir(dirname + dataFolderPath)
print "CurrentFolder(This Should be 'Resources/MillionSongSubset/data' folder): ", os.getcwd()

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

#Put feature functions in a list in order
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


#Initialize Feature Matrix Data Structure with feature names as header, (going to be list of lists)
featureMatrix = []
print "feature_names length", len(feature_names)

#Start the time
startTime = time.time()
#Start going into files
flag = 1
numberOf_h5Files = 0
for root, dirs, files in os.walk(os.getcwd()):
    for name in files:
        if name.endswith(".h5"):
            #increment to count the number of .h5 files
            numberOf_h5Files += 1

            #open the h5 file
            h5SongFile = hdf5_getters.open_h5_file_read(os.path.join(root,name))

            #create the sample row for feature matrix
            #initialize the list with track ID and song ID
            sampleRow = [hdf5_getters.get_track_id(h5SongFile),
                         hdf5_getters.get_song_id(h5SongFile)]

            cnt = 0
            for func in featureFunctions:
                cnt += 1
                #print "functionNumber: ", cnt
                #print "BarStrctr: ", hdf5_getters.get_bars_start(h5SongFile)
                sampleRow += func(h5SongFile)

            #Construct the feature matrix that contains all the songs...
            #Data structure for featureMatrix will be list of lists, so that...
            #Numpy will be able to convert it to a matrix in a way that...
            #All list elements will become rows
            featureMatrix += [sampleRow]

            #Print out rows
            print "SongNumber: ", numberOf_h5Files, " F: ", sampleRow

            #close the file
            h5SongFile.close()
        else:
            #place holder for non .h5 files
            pass


#Make it numpy matrix
featureMatrixNumpy = numpy.array([feature_names] + featureMatrix)
#Save the feature matrix to file
#cd into Extracted_Data folder
extractedDataFolderPath = "/Extracted_Data"
os.chdir(dirname + extractedDataFolderPath)
print "CurrentDirectory(This should be '/Extracted_Data' folder): ", os.getcwd()
print "Saving the featureMatrix.bin file..."
#Save the featureMatrix.bin file
numpy.savetxt("featureMatrix.bin", featureMatrixNumpy, delimiter='|',fmt="%s")
print "Process Complete."
endTime = time.time()
print "Number Of h5 files: ", numberOf_h5Files
print "FeatureMatrix (Shape)", featureMatrixNumpy.shape
print "FeatureMatrix (Rows): ", featureMatrixNumpy.shape[0], "(Columns): ", featureMatrixNumpy.shape[1]
print "Elapsed Time: ", endTime - startTime, "Seconds"