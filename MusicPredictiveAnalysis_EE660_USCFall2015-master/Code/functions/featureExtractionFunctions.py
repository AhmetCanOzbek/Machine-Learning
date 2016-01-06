__author__ = "Can Ozbek"

import hdf5_getters
import numpy

#Continuous variable feature functions
def getBarDuration(h5):
    #Returns the average duration of bars in a song
    barsVector = hdf5_getters.get_bars_start(h5)
    #If there is no information, return None
    if len(barsVector) < 2:
        return ["nan"]

    #get the sum
    barDurationSums = 0
    for i in range(0,barsVector.size - 1):
        barDurationSums += (barsVector[i + 1] - barsVector[i])

    #get the average duration
    avgBarDuration = barDurationSums / barsVector.size
    return [avgBarDuration]

def getLoudness(h5):
    #Returns the loudness of the song
    return [hdf5_getters.get_loudness(h5)]

def getTempo(h5):
    #Returns the tempo of the song
    return [hdf5_getters.get_tempo(h5)]

def getArtistFamiliarity(h5):
    #Returns the artist familiarity value
    return [hdf5_getters.get_artist_familiarity(h5)]

def getArtistHotttnesss(h5):
    #Returns the artist hotttnesss
    return [hdf5_getters.get_artist_hotttnesss(h5)]

def getSongHotttnesss(h5):
    #Returns song hotttnesss
    return [hdf5_getters.get_song_hotttnesss(h5)]

def getYear(h5):
    #Returns song year
    year = hdf5_getters.get_year(h5)
    if year == 0:
        return ["nan"]
    return [year]

def getSegmentPitchesMean(h5):
    """
    Returns the mean of all pitches from all segments
    :param h5: input song file
    :return: 12 dimensional list for all pitches
    """
    pitches = hdf5_getters.get_segments_pitches(h5)
    confidence = hdf5_getters.get_segments_confidence(h5)
    is_confident = numpy.array(confidence)>0.5
    confident_feature = pitches[is_confident,:]
    expanded_feat_mean = numpy.mean(confident_feature, axis=0)
    return expanded_feat_mean.tolist()


def getSegmentPitchesVar(h5):
    """
    Returns the variance of all pitches from all segments
    :param h5: input song file
    :return: 12 dimensional list for all pitches
    """
    pitches = hdf5_getters.get_segments_pitches(h5)
    confidence = hdf5_getters.get_segments_confidence(h5)
    is_confident = numpy.array(confidence)>0.5
    confident_feature = pitches[is_confident,:]
    expanded_feat_var = numpy.var(confident_feature, axis=0)
    return expanded_feat_var.tolist()

def getSegmentTimbreMean(h5):
    """
    Returns the mean of each timbre from all segments
    :param h5: input song file
    :return: 12 dimensional list
    """
    timbre = hdf5_getters.get_segments_timbre(h5)
    expanded_feat_mean = numpy.mean(timbre, axis=0)
    return expanded_feat_mean.tolist()

def getSegmentTimbreVar(h5):
    """
    Returns the variance of each timbre from all segments
    :param h5: input song file
    :return: 12 dimensional list
    """
    timbre = hdf5_getters.get_segments_timbre(h5)
    expanded_feat_var = numpy.var(timbre, axis=0)
    return expanded_feat_var.tolist()



#Categorical Feature Functions:
#TODO: -1 No Value Issues

def getMode(h5):
    """
    Returns whether the song is in major or minor key
    #Categorical Feature
    #0 --> [0 1]
    #1 --> [1 0]
    ::return: 2 dimensional list
    """
    mode = hdf5_getters.get_mode(h5)
    if mode == 0:
        return [0,1]
    elif mode == 1:
        return [1,0]

    #Mode info not available
    return ["nan"]

def getKey(h5):
    """
    Returns the key of the song as an expanded feature
    As there are 12 keys, the output vector is going to be 12 dimensional
    :param h5: input song file
    :return: 12 dimensional vector (list)
    Examples:
        Key: 1 --> Output: [1 0 0 0 0 0 0 0 0 0 0 0]
        Key 11 --> Output: [0 0 0 0 0 0 0 0 0 0 1 0]
        Key 12 --> Output: [0 0 0 0 0 0 0 0 0 0 0 1]
    """
    #initializing the 12 dimensional vector as a list of zeros
    output_vector = [0] * 12
    #get key
    key =  hdf5_getters.get_key(h5)
    #Key info not available, return None, exit function
    if key == -1:
        return ["nan"]

    #put the '1' into the appropriate index
    output_vector[key] = 1
    return output_vector


def getTimeSignature(h5):
    """
    Returns the Time Signature of the information as a 6 dimensional vector.
    The time signature ranges from 3 to 7 indicating time signatures of 3/4 to 7/4
    A value of 1 indicates a rather complex or changing time signature
    This is a categorical feature and we have (3 to 7) and (1) = 6 different values
    Examples
        Time Signature = 1 --> Output: [1 0 0 0 0 0]
        Time Signature = 3 --> Output: [0 1 0 0 0 0]
        Time Signature = 4 --> Output: [0 0 1 0 0 0]
        ...
        Time Signature = 7 --> Output: [0 0 0 0 0 1]
    :param h5: input song file
    :return: Six dimensional vector as a list
    """
    timeSignature = hdf5_getters.get_time_signature(h5)
    #If Time Signature Information not available return missing data
    if timeSignature == -1:
        return ["nan"]

    if timeSignature == 1:
        return [1,0,0,0,0,0]
    else:
        #initialize a list of length 6 with zeros
        outputVector = [0] * 6
        outputVector[timeSignature-2] = 1
        return outputVector





