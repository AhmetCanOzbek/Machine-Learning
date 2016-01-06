__author__ = 'Arnav & Nishant'

import pandas
import numpy
import time

start_time=time.time()

import pandas
import numpy
import time


col_meta = ['track_id','song_id','wc_genre', 'wc_year']

col_data_in = [
        'track_id', 'song_id',
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


col_data_out = [ 'wc_genre', 'track_id',

                 'AvgBarDuration','Loudness', 'Tempo','ArtistFamiliarity','ArtistHotttnesss','SongHotttnesss',
                 'Mode[0]','Mode[1]','wc_year',

                 'Key[0]','Key[1]','Key[2]','Key[3]','Key[4]','Key[5]',
                 'Key[6]','Key[7]','Key[8]','Key[9]','Key[10]','Key[11]',

                 'PicthesMean[0]','PicthesMean[1]','PicthesMean[2]','PicthesMean[3]','PicthesMean[4]','PicthesMean[5]',
                 'PicthesMean[6]','PicthesMean[7]','PicthesMean[8]','PicthesMean[9]','PicthesMean[10]','PicthesMean[11]',

                 'PitchesVar[0]','PitchesVar[1]','PitchesVar[2]','PitchesVar[3]','PitchesVar[4]','PitchesVar[5]',
                 'PitchesVar[6]','PitchesVar[7]','PitchesVar[8]','PitchesVar[9]','PitchesVar[10]','PitchesVar[11]',

                 'TimbreMean[0]','TimbreMean[1]','TimbreMean[2]','TimbreMean[3]','TimbreMean[4]','TimbreMean[5]',
                 'TimbreMean[6]','TimbreMean[7]','TimbreMean[8]','TimbreMean[9]','TimbreMean[10]','TimbreMean[11]',

                 'TimbreVar[0]','TimbreVar[1]','TimbreVar[2]','TimbreVar[3]','TimbreVar[4]','TimbreVar[5]',
                 'TimbreVar[6]','TimbreVar[7]','TimbreVar[8]','TimbreVar[9]','TimbreVar[10]','TimbreVar[11]',
                 'TimeSig[0]', 'TimeSig[1]', 'TimeSig[2]', 'TimeSig[3]', 'TimeSig[4]', 'TimeSig[5]']


start_time=time.time()

df_meta = pandas.read_csv('MSD_Data_Genre_WikiCorrected_DupRemoved.bin', header=None, delimiter = "|", names=col_meta)

df_data = pandas.read_csv('feature_matrix_full.bin', header=0, delimiter = "|", names=col_data_in)


df_merged = pandas.merge(df_data, df_meta, how='left', on=['track_id', 'track_id', 'song_id', 'song_id'])

#handling missing data
for index, row in df_merged.iterrows():
    if (row['wc_genre'] == '') or (row['wc_genre'] is None):
        df_merged.set_value(index, 'wc_genre', 'UNCAT')

    if (not numpy.isfinite(row['wc_year'])) or (row['wc_year'] == '') or (row['wc_year'] is None):
        df_merged.set_value(index, 'wc_year', 0)


elapsed_time = time.time() - start_time
print "elapsed time : ", elapsed_time

df_merged.to_csv('MSD_Final_Dataset_For_EE660.bin', sep="|", index=False, header=None, columns=col_data_out)


elapsed_time = time.time() - start_time
print "elapsed time : ", elapsed_time