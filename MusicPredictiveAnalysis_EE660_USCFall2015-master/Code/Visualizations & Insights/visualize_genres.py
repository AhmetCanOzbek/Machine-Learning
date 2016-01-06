__author__ = 'Arnav'

import numpy as np
import matplotlib.pyplot as plt
import pandas

df = pandas.read_csv('find_freq_genre_MapReduce_out.bin', header=None, delimiter="\t")

labels = df[0].tolist()
counts = df[1].tolist()

fig, ax = plt.subplots()
width = 0.5
ind = np.arange(10)
rect = ax.bar(ind, counts, width, color='r')
ax.set_xlim(-width, len(ind)+width)
ax.set_ylabel('Number of Songs in dataset')
ax.set_title('Count of songs per genre in dataset')
ax.set_xticks(ind+width/2)
xtickNames = ax.set_xticklabels(labels)
plt.setp(xtickNames, rotation=45, fontsize=10)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height), ha='center', va='bottom')
autolabel(rect)
plt.show()
