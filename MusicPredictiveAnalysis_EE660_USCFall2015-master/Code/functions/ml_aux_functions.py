__author__ = "Can Ozbek"

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def getUniqueCount(df_column):
    """
    Returns a dictionary of unique counts
    :param df_column: pandas series, (column)
    :return: dictionary containing unique counts
    """
    unique_values_list = df_column.unique().tolist()
    unique_count_dict = dict.fromkeys(unique_values_list)
    for value in unique_values_list:
        unique_count_dict[value] = sum(df_column == value)
    return unique_count_dict


def get_confusion_matrix(y_true, y_predicted):
    #Get the Class Labels
    classLabels = y_true.unique().tolist()
    classLabels.sort()

    print "ClassLabels: ", classLabels
    #Get the confusion matrix
    cmatrix = confusion_matrix(y_true, y_predicted, classLabels)
    return cmatrix


def plot_confusion_matrix(y_true, y_predicted, title='Confusion matrix'):
    """
    Returns the confusion matrix as a numpy array
    """
    #Get the Class Labels
    classLabels = y_true.unique().tolist()
    classLabels.sort()
    #Get the confusion matrix
    cmatrix = confusion_matrix(y_true, y_predicted, classLabels)
    #Plot the figure
    plt.figure()
    #Ticks
    # Keep major ticks labeless
    plt.xticks(range(len(classLabels)+1), [])
    plt.yticks(range(len(classLabels)+1), [])
    # Place labels on minor ticks
    plt.gca().set_xticks([x + 0.5 for x in range(len(classLabels))], minor=True)
    plt.gca().set_xticklabels(classLabels, rotation='45', fontsize=10, minor=True)
    plt.gca().set_yticks([y + 0.5 for y in range(len(classLabels))], minor=True)
    plt.gca().set_yticklabels(classLabels[::-1], fontsize=10, minor=True)
    # Finally, hide minor tick marks...
    plt.gca().tick_params('both', width=0, which='minor')

    #Grid on
    plt.grid(True)
    #Put the values into the plot
    for x in range(cmatrix.shape[0]):
        for y in range(cmatrix.shape[1]):
            if x==cmatrix.shape[0]-(y+1):
                plt.text(x+0.5,y+0.5,cmatrix[cmatrix.shape[0]-(y+1)][x],
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        color="Green",
                                        fontsize = 15)
            elif cmatrix[cmatrix.shape[0]-(y+1)][x] > 0:
                plt.text(x+0.5,y+0.5,cmatrix[cmatrix.shape[0]-(y+1)][x],
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        color="Red",
                                        fontsize = 15)
            else:
                plt.text(x+0.5,y+0.5,cmatrix[cmatrix.shape[0]-(y+1)][x],
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        fontsize = 15)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(title)
    plt.tight_layout()
    return cmatrix

def plot_histogram(d):
    plt.figure()
    X = np.arange(len(d))
    plt.bar(X, d.values(), align='center', width=0.5)
    plt.xticks(X, d.keys())
    ymax = max(d.values()) + 1
    plt.ylim(0, ymax)
    plt.tight_layout()
    return d

def get_error_rate(y_true, y_predicted):
    return sum(y_true != y_predicted) / float(y_true.shape[0])




