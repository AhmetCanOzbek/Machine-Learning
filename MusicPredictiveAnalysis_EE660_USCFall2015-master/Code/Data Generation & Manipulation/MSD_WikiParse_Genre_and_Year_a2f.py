# !/usr/bin/env python

__author__ = 'NishantNath'


'''
Using : Python 2.7+ (backward compatibility exists for Python 3.x if separate environment created)
Required files : none
Required packages : numpy, time, wikipedia, rdflib, requests, unicodedata, opeartor, collections, pandas

1.

# This script extracts the album name for track from file, parses wiki api for wiki url then extracts genre & release date for track

'''

import numpy
import time
import wikipedia
from rdflib import Graph, URIRef
from collections import Counter
import operator
import requests
import unicodedata
import pandas


if __name__ == '__main__':
    print('----- started -----')

    # global variables declared here
    my_genre = ['CLASSICAL','METAL','HIPHOP','DANCE','JAZZ','FOLK','SOUL','ROCK','POP','BLUES']

    start_time=time.time()

    col_input=['track_id', 'song_id', 'album_name', 'year']
    df_input = pandas.read_csv('wiki_parse_a2f.bin', header=None, delimiter = "|", names=col_input)

    process_completion = 0
    counter = 0
    file_io_counter = 0
    for index, row in df_input.iterrows():
        if process_completion % 1000 == 0:
            print "done :", process_completion/1000.0, "%"
            process_completion += 1
        else:
            process_completion += 1

        album_name = row['album_name']
        album_name = str(album_name).replace(","," ").replace("'","").replace("-"," ").replace("("," ").replace(")"," ").replace("/"," ").replace("\\"," ")
        genre = 'UNCAT'
        extracted_date = '0'

        #wiki parsing starts here
        try:
            wiki_page = wikipedia.page(album_name)
            wiki_page_url = str(unicodedata.normalize('NFKD', wiki_page.url).encode('ASCII', 'ignore'))
            # print type(wiki_page.url)
        except:
            # print "Not a valid wiki page for : ", row['track_id']
            pass
        else:
            url_to_parse = "http://dbpedia.org/page/" + wiki_page_url.split('/')[-1]

            if requests.head(url_to_parse).status_code == 200:

                # print url_to_parse
                g = Graph()
                g.parse(url_to_parse)

                for ontology_branch_date in g.subject_objects(URIRef("http://dbpedia.org/ontology/releaseDate")):
                    extracted_date = int(str(ontology_branch_date[1].split('-')[0]))

                my_list =[]
                for ontology_branch in g.subject_objects(URIRef("http://dbpedia.org/ontology/genre")):
                    my_list.append(ontology_branch)

                tags=[]
                strings = [str(unicodedata.normalize('NFKD', x[1]).encode('ASCII', 'ignore')) for x in my_list]
                for i in range(0,len(strings)):
                    tags.append(strings[i].split('/')[-1].replace("HIP HOP","HIPHOP").replace("HIP-HOP","HIPHOP").replace("BLUE","BLUES").replace("BLUESS","BLUES").split('_'))

                words = [item for sublist in tags for item in sublist]
                words = filter(None,words)
                words = filter(lambda x:x != '', words)
                if len(words) != 0:
                    cap_words = [word.upper() for word in words]
                    word_counts = Counter(cap_words)
                    genre = max(word_counts.iteritems(), key=operator.itemgetter(1))[0]

                    # fixes needed but later maybe?
                    if genre not in my_genre or genre == '':
                        genre = 'UNCAT'

                del words
                del strings
                del tags
                del my_list
                del g

                counter += 1
                file_io_counter += 1
                if wiki_page.url == '' or wiki_page.url is None:
                    page = ''
                else:
                    page = wiki_page_url

                if file_io_counter % 50 == 0:
                    if counter == 1:
                        my_array_data_extract_wiki_a2f = numpy.array([row['track_id'], row['song_id'], page, album_name, extracted_date, genre])
                    else :
                        my_array_data_extract_wiki_a2f = numpy.vstack((my_array_data_extract_wiki_a2f,numpy.array([row['track_id'], row['song_id'], page, album_name, extracted_date, genre])))

                    f_handle = file('MSD_Data_Extract_Wiki_a2f.bin','a')
                    numpy.savetxt(f_handle, my_array_data_extract_wiki_a2f, delimiter='|',fmt='%s')
                    f_handle.close()

                    del my_array_data_extract_wiki_a2f
                    counter = 0
                    file_io_counter = 0
                else:
                    if counter == 1:
                        my_array_data_extract_wiki_a2f = numpy.array([row['track_id'], row['song_id'], page, album_name, extracted_date, genre])
                    else :
                        my_array_data_extract_wiki_a2f = numpy.vstack((my_array_data_extract_wiki_a2f,numpy.array([row['track_id'], row['song_id'], page, album_name, extracted_date, genre])))
            else:
                print "Not A Valid DBPEDIA URL for : ",wiki_page_url
                print "Track id for above : ", row['track_id']


    elapsed_time = time.time() - start_time
    print("elapsed time : ",elapsed_time)

    try:
        my_array_data_extract_wiki_a2f
    except NameError:
        print "my_array_data_extract_wiki_a2f does not exist currently"
    else:
        f_handle = file('MSD_Data_Extract_Wiki_a2f.bin','a')
        numpy.savetxt(f_handle, my_array_data_extract_wiki_a2f, delimiter='|',fmt='%s')
        f_handle.close()

    print('----- done -----')