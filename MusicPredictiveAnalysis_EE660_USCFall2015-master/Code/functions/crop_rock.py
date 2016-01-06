'''
Usage :
input = pickle.load(open("msd_train_t1.pkl", "rb"))
# print input.shape[0]

maxval = crop_rock.find_second_max_value(input)
# print maxval

filtered = crop_rock.drop_excess_rows(input, maxval)
# print filtered.shape[0]

'''


import numpy
import pandas

def find_second_max_value(input):
    counter_CLASSICAL = 0; counter_METAL = 0; counter_HIPHOP = 0; counter_DANCE = 0; counter_JAZZ = 0;
    counter_FOLK = 0; counter_SOUL = 0; counter_ROCK = 0; counter_POP = 0; counter_BLUES = 0

    for index, row in input.iterrows():
        if row['Genre'] == 'CLASSICAL':
            counter_CLASSICAL += 1
        if row['Genre'] == 'METAL':
            counter_METAL += 1
        if row['Genre'] == 'HIPHOP':
            counter_HIPHOP += 1
        if row['Genre'] == 'DANCE':
            counter_DANCE += 1
        if row['Genre'] == 'JAZZ':
            counter_JAZZ += 1
        if row['Genre'] == 'FOLK':
            counter_FOLK += 1
        if row['Genre'] == 'SOUL':
            counter_SOUL += 1
        if row['Genre'] == 'ROCK':
            counter_ROCK += 1
        if row['Genre'] == 'POP':
            counter_POP += 1
        if row['Genre'] == 'BLUES':
            counter_BLUES += 1

    arr = numpy.array([counter_CLASSICAL,counter_METAL,counter_HIPHOP,counter_DANCE,counter_JAZZ,counter_FOLK,counter_SOUL,counter_ROCK,counter_POP,counter_BLUES])
    ind = numpy.argpartition(arr, -2)[-2:]
    maxval = min(arr[ind])
    return maxval


def drop_excess_rows(input, maxval):
    counter_CLASSICAL = 0; counter_METAL = 0; counter_HIPHOP = 0; counter_DANCE = 0; counter_JAZZ = 0;
    counter_FOLK = 0; counter_SOUL = 0; counter_ROCK = 0; counter_POP = 0; counter_BLUES = 0

    drop_index=[]
    for index, row in input.iterrows():
        # if row['Genre'] == 'CLASSICAL':
        #     counter_CLASSICAL += 1
        #     if counter_CLASSICAL <= maxval:
        #         pass
        #     else:
        #         drop_index.append(index)
        # if row['Genre'] == 'METAL':
        #     counter_METAL += 1
        #     if counter_METAL <= maxval:
        #         pass
        #     else:
        #         drop_index.append(index)
        # if row['Genre'] == 'HIPHOP':
        #     counter_HIPHOP += 1
        #     if counter_HIPHOP <= maxval:
        #         pass
        #     else:
        #         drop_index.append(index)
        # if row['Genre'] == 'DANCE':
        #     counter_DANCE += 1
        #     if counter_DANCE <= maxval:
        #         pass
        #     else:
        #         drop_index.append(index)
        # if row['Genre'] == 'JAZZ':
        #     counter_JAZZ += 1
        #     if counter_JAZZ <= maxval:
        #         pass
        #     else:
        #         drop_index.append(index)
        # if row['Genre'] == 'FOLK':
        #     counter_FOLK += 1
        #     if counter_FOLK <= maxval:
        #         pass
        #     else:
        #         drop_index.append(index)
        # if row['Genre'] == 'SOUL':
        #     counter_SOUL += 1
        #     if counter_SOUL <= maxval:
        #         pass
        #     else:
        #         drop_index.append(index)
        if row['Genre'] == 'ROCK':
            counter_ROCK += 1
            if counter_ROCK <= maxval:
                pass
            else:
                drop_index.append(index)
        # if row['Genre'] == 'POP':
        #     counter_POP += 1
        #     if counter_POP <= maxval:
        #         pass
        #     else:
        #         drop_index.append(index)
        # if row['Genre'] == 'BLUES':
        #     counter_BLUES += 1
        #     if counter_BLUES <= maxval:
        #         pass
        #     else:
        #         drop_index.append(index)

    filtered = input.drop(drop_index)

    return filtered
