# !/usr/bin/env python

__author__ = 'NishantNath'

'''
Using : Python 2.7+
Required files : none
Required packages : mrjob

# Uses Map-Reduce to find the counts of song id for MSD dataset
# Usage : python find_freq_genre_MapReduce.py find_freq_genres_testfile.bin > find_freq_genre_MapReduce_out.bin
'''

from mrjob.job import MRJob

class MR_find_freq_songID(MRJob):

    def mapper(self, _, line):
        (_,_,genre,year) = line.split('|')
        yield genre, 1

    def reducer(self, key, values):
        yield key, sum(values)


if __name__ == '__main__':
    MR_find_freq_songID.run()