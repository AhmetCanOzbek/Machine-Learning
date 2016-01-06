__author__ = 'Arnav'
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv

#############Read the years from txt file################
year_list = [];
f= open('../Resources/FullSet/AdditionalFiles/tracks_per_year.txt', 'rU', encoding='utf8')
data=csv.reader(f)
for row in data:
    line = row[0]
    custom_format = line.split('<SEP>')
    year = custom_format[0]
    #year = year.encode('utf8')
    year_list.append(year)
f.close()

count = 0
ignore = 0
init = 1922

"""
for current_year in year_list:
    #print(current_year)
    if int(current_year) == init:
        count += 1
        previous_year = current_year
    else:
        print("Year: " + str(previous_year) + " Count: " + str(count))
        count = 1
        init = int(current_year)
"""
#############Ignore the years not in proper format and count################
f_out = open('year_vs_count.txt', 'a')
for current_year in year_list:
    try:
        if int(current_year) == init:
            count += 1
            previous_year = current_year
        else:
            to_write = str(previous_year) + "|" + str(count) + "\n"
            f_out.write(to_write)
            count = 1
            init = int(current_year)
    except:
        ignore +=1
print(str(ignore)+" Years not in correct format, ignored")



