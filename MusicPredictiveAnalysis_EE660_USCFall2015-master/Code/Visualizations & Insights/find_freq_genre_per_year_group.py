# !/usr/bin/env python

__author__ = 'NishantNath'

'''
Using : Python 2.7+
Required files : none
Required packages : pandas, numpy, time

# Counts the number of files in each genre per years-period
'''

import pandas
import numpy
import time

start_time=time.time()

col_meta = ['song_id', 'track_id', 'genre', 'year']
df_meta = pandas.read_csv('find_freq_genres_testfile.bin', header=None, delimiter = "|", names=col_meta)

df1 = df_meta[df_meta['year']>=1921][df_meta['year']<=1963]
df2 = df_meta[df_meta['year']>=1964][df_meta['year']<=1980]
df3 = df_meta[df_meta['year']>=1981][df_meta['year']<=1992]
df4 = df_meta[df_meta['year']>=1993][df_meta['year']<=2002]
df5 = df_meta[df_meta['year']>=2003]

f_out = open('find_freq_genres_out.bin', 'a')

# DF1
counter_CLASSICAL = 0
counter_METAL = 0
counter_HIPHOP = 0
counter_DANCE = 0
counter_JAZZ = 0
counter_FOLK = 0
counter_SOUL = 0
counter_ROCK = 0
counter_POP = 0
counter_BLUES = 0

for index, row in df1.iterrows():
    if row['genre'] == 'CLASSICAL':
        counter_CLASSICAL += 1
    if row['genre'] == 'METAL':
        counter_METAL += 1
    if row['genre'] == 'HIPHOP':
        counter_HIPHOP += 1
    if row['genre'] == 'DANCE':
        counter_DANCE += 1
    if row['genre'] == 'JAZZ':
        counter_JAZZ += 1
    if row['genre'] == 'FOLK':
        counter_FOLK += 1
    if row['genre'] == 'SOUL':
        counter_SOUL += 1
    if row['genre'] == 'ROCK':
        counter_ROCK += 1
    if row['genre'] == 'POP':
        counter_POP += 1
    if row['genre'] == 'BLUES':
        counter_BLUES += 1

f_out.write("DF1"+ "\n")
f_out.write("counter_CLASSICAL : " + str(counter_CLASSICAL)+ "\n")
f_out.write("counter_METAL : " + str(counter_METAL)+ "\n")
f_out.write("counter_HIPHOP : " + str(counter_HIPHOP)+ "\n")
f_out.write("counter_DANCE : " + str(counter_DANCE)+ "\n")
f_out.write("counter_JAZZ : " + str(counter_JAZZ)+ "\n")
f_out.write("counter_FOLK : " + str(counter_FOLK)+ "\n")
f_out.write("counter_SOUL : " + str(counter_SOUL)+ "\n")
f_out.write("counter_ROCK : " + str(counter_ROCK)+ "\n")
f_out.write("counter_POP : " + str(counter_POP)+ "\n")
f_out.write("counter_BLUES : " + str(counter_BLUES)+ "\n")

#DF2
counter_CLASSICAL = 0
counter_METAL = 0
counter_HIPHOP = 0
counter_DANCE = 0
counter_JAZZ = 0
counter_FOLK = 0
counter_SOUL = 0
counter_ROCK = 0
counter_POP = 0
counter_BLUES = 0

for index, row in df2.iterrows():
    if row['genre'] == 'CLASSICAL':
        counter_CLASSICAL += 1
    if row['genre'] == 'METAL':
        counter_METAL += 1
    if row['genre'] == 'HIPHOP':
        counter_HIPHOP += 1
    if row['genre'] == 'DANCE':
        counter_DANCE += 1
    if row['genre'] == 'JAZZ':
        counter_JAZZ += 1
    if row['genre'] == 'FOLK':
        counter_FOLK += 1
    if row['genre'] == 'SOUL':
        counter_SOUL += 1
    if row['genre'] == 'ROCK':
        counter_ROCK += 1
    if row['genre'] == 'POP':
        counter_POP += 1
    if row['genre'] == 'BLUES':
        counter_BLUES += 1

f_out.write("DF2"+ "\n")
f_out.write("counter_CLASSICAL : " + str(counter_CLASSICAL)+ "\n")
f_out.write("counter_METAL : " + str(counter_METAL)+ "\n")
f_out.write("counter_HIPHOP : " + str(counter_HIPHOP)+ "\n")
f_out.write("counter_DANCE : " + str(counter_DANCE)+ "\n")
f_out.write("counter_JAZZ : " + str(counter_JAZZ)+ "\n")
f_out.write("counter_FOLK : " + str(counter_FOLK)+ "\n")
f_out.write("counter_SOUL : " + str(counter_SOUL)+ "\n")
f_out.write("counter_ROCK : " + str(counter_ROCK)+ "\n")
f_out.write("counter_POP : " + str(counter_POP)+ "\n")
f_out.write("counter_BLUES : " + str(counter_BLUES)+ "\n")


#DF3
counter_CLASSICAL = 0
counter_METAL = 0
counter_HIPHOP = 0
counter_DANCE = 0
counter_JAZZ = 0
counter_FOLK = 0
counter_SOUL = 0
counter_ROCK = 0
counter_POP = 0
counter_BLUES = 0

for index, row in df3.iterrows():
    if row['genre'] == 'CLASSICAL':
        counter_CLASSICAL += 1
    if row['genre'] == 'METAL':
        counter_METAL += 1
    if row['genre'] == 'HIPHOP':
        counter_HIPHOP += 1
    if row['genre'] == 'DANCE':
        counter_DANCE += 1
    if row['genre'] == 'JAZZ':
        counter_JAZZ += 1
    if row['genre'] == 'FOLK':
        counter_FOLK += 1
    if row['genre'] == 'SOUL':
        counter_SOUL += 1
    if row['genre'] == 'ROCK':
        counter_ROCK += 1
    if row['genre'] == 'POP':
        counter_POP += 1
    if row['genre'] == 'BLUES':
        counter_BLUES += 1

f_out.write("DF3"+ "\n")
f_out.write("counter_CLASSICAL : " + str(counter_CLASSICAL)+ "\n")
f_out.write("counter_METAL : " + str(counter_METAL)+ "\n")
f_out.write("counter_HIPHOP : " + str(counter_HIPHOP)+ "\n")
f_out.write("counter_DANCE : " + str(counter_DANCE)+ "\n")
f_out.write("counter_JAZZ : " + str(counter_JAZZ)+ "\n")
f_out.write("counter_FOLK : " + str(counter_FOLK)+ "\n")
f_out.write("counter_SOUL : " + str(counter_SOUL)+ "\n")
f_out.write("counter_ROCK : " + str(counter_ROCK)+ "\n")
f_out.write("counter_POP : " + str(counter_POP)+ "\n")
f_out.write("counter_BLUES : " + str(counter_BLUES)+ "\n")

#DF4
counter_CLASSICAL = 0
counter_METAL = 0
counter_HIPHOP = 0
counter_DANCE = 0
counter_JAZZ = 0
counter_FOLK = 0
counter_SOUL = 0
counter_ROCK = 0
counter_POP = 0
counter_BLUES = 0

for index, row in df4.iterrows():
    if row['genre'] == 'CLASSICAL':
        counter_CLASSICAL += 1
    if row['genre'] == 'METAL':
        counter_METAL += 1
    if row['genre'] == 'HIPHOP':
        counter_HIPHOP += 1
    if row['genre'] == 'DANCE':
        counter_DANCE += 1
    if row['genre'] == 'JAZZ':
        counter_JAZZ += 1
    if row['genre'] == 'FOLK':
        counter_FOLK += 1
    if row['genre'] == 'SOUL':
        counter_SOUL += 1
    if row['genre'] == 'ROCK':
        counter_ROCK += 1
    if row['genre'] == 'POP':
        counter_POP += 1
    if row['genre'] == 'BLUES':
        counter_BLUES += 1

f_out.write("DF4"+ "\n")
f_out.write("counter_CLASSICAL : " + str(counter_CLASSICAL)+ "\n")
f_out.write("counter_METAL : " + str(counter_METAL)+ "\n")
f_out.write("counter_HIPHOP : " + str(counter_HIPHOP)+ "\n")
f_out.write("counter_DANCE : " + str(counter_DANCE)+ "\n")
f_out.write("counter_JAZZ : " + str(counter_JAZZ)+ "\n")
f_out.write("counter_FOLK : " + str(counter_FOLK)+ "\n")
f_out.write("counter_SOUL : " + str(counter_SOUL)+ "\n")
f_out.write("counter_ROCK : " + str(counter_ROCK)+ "\n")
f_out.write("counter_POP : " + str(counter_POP)+ "\n")
f_out.write("counter_BLUES : " + str(counter_BLUES)+ "\n")

#DF5
counter_CLASSICAL = 0
counter_METAL = 0
counter_HIPHOP = 0
counter_DANCE = 0
counter_JAZZ = 0
counter_FOLK = 0
counter_SOUL = 0
counter_ROCK = 0
counter_POP = 0
counter_BLUES = 0

for index, row in df5.iterrows():
    if row['genre'] == 'CLASSICAL':
        counter_CLASSICAL += 1
    if row['genre'] == 'METAL':
        counter_METAL += 1
    if row['genre'] == 'HIPHOP':
        counter_HIPHOP += 1
    if row['genre'] == 'DANCE':
        counter_DANCE += 1
    if row['genre'] == 'JAZZ':
        counter_JAZZ += 1
    if row['genre'] == 'FOLK':
        counter_FOLK += 1
    if row['genre'] == 'SOUL':
        counter_SOUL += 1
    if row['genre'] == 'ROCK':
        counter_ROCK += 1
    if row['genre'] == 'POP':
        counter_POP += 1
    if row['genre'] == 'BLUES':
        counter_BLUES += 1

f_out.write("DF5"+ "\n")
f_out.write("counter_CLASSICAL : " + str(counter_CLASSICAL)+ "\n")
f_out.write("counter_METAL : " + str(counter_METAL)+ "\n")
f_out.write("counter_HIPHOP : " + str(counter_HIPHOP)+ "\n")
f_out.write("counter_DANCE : " + str(counter_DANCE)+ "\n")
f_out.write("counter_JAZZ : " + str(counter_JAZZ)+ "\n")
f_out.write("counter_FOLK : " + str(counter_FOLK)+ "\n")
f_out.write("counter_SOUL : " + str(counter_SOUL)+ "\n")
f_out.write("counter_ROCK : " + str(counter_ROCK)+ "\n")
f_out.write("counter_POP : " + str(counter_POP)+ "\n")
f_out.write("counter_BLUES : " + str(counter_BLUES)+ "\n")




f_out.close()