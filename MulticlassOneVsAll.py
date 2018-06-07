import Initializing
import Plot
import LogisticRegression
import Perceptron
import numpy as np
import matplotlib.markers as mark
import matplotlib.pyplot as plt
import random
import copy

def processDataPerClass(t, classId, algorithm):
    if algorithm == 'perceptron':
        negClass = -1
    else:
        negClass = 0
    tmpT = copy.copy(t)
    # podaci cije su labele jednake prosledjenoj klasi u logistickoj regresiji dobijaju vrednost 1, u suprotnom 0
    for tn in range(len(t[0])):
        if t[0][tn] == classId:
            tmpT[0][tn] = 1
        else:
            tmpT[0][tn] = negClass
    return tmpT

def trainClassifiers(x, t, algorithm):
    labels = np.unique(t)
    listW = []
    listB = []
    for i in range(len(labels)):
        tmpT = processDataPerClass(t, labels[i], algorithm)
        w, b = Initializing.initialParam(x)
        if algorithm == 'perceptron':

            w, b = Perceptron.train(x, tmpT, w, b)
        else:
            w, b = LogisticRegression.trein(w, b, x, tmpT)
        listW.append(np.array(w[0]).reshape(1,len(w[0])))
        listB.append(np.array(b[0]).reshape(1,1))
    return [listW, listB]

def oneVsAllLogisticRegressionExample():
    data = [[3.4, 5.9, 2], [2.7, 7.9, 2], [3.8, 7.8, 2], [3, 6, 2], [3.5, 8, 2],
            [3.7, 7.5, 2], [4, 2, 1], [1, 1, 0], [2, 2, 0], [1, 3, 0], [2.7, 2.4, 0],
            [3, 2, 0], [5, 1, 1], [5, 2, 1], [6, 2, 1], [6, 3, 1], [6, 5, 1]]

    x, t = Initializing.processData(data)  # iz pocetnog skupa podataka razdvojimo podatke i labele
    Plot.plotData(x, t)  # nacrtamo podatke
    w, b = trainClassifiers(x, t, "logistic")

    Plot.plotLinesMulticlassOneVsAll(x, t, w, b)
    plt.show()

def oneVsAllPerceptronExample():
    data = [[3.4, 5.9, 2], [2.7, 7.9, 2], [3.8, 7.8, 2], [3, 6, 2], [3.5, 8, 2],
            [3.7, 7.5, 2], [4, 2, 1], [1, 1, 0], [2, 2, 0], [1, 3, 0], [2.7, 2.4, 0],
            [3, 2, 0], [5, 1, 1], [5, 2, 1], [6, 2, 1], [6, 3, 1], [6, 5, 1]]

    x, t = Initializing.processData(data)  # iz pocetnog skupa podataka razdvojimo podatke i labele
    #Plot.plotData(x, t)  # nacrtamo podatke
    w, b = trainClassifiers(x, t, "perceptron")

    fig = Plot.plotLinesMulticlassOneVsAll(x, t, w, b)
    return fig
    #plt.show()

#oneVsAllLogisticRegressionExample()
#oneVsAllPerceptronExample()

# def plot_line(x_kord, w, b):
#     #x_kord = [0, 4]
#     y_kord = [-(w[0,0]/w[0,1])*i - (b[0,0]/w[0,1]) for i in x_kord]
#     plt.plot(x_kord, y_kord)

# plot_line([0,4], w[0]-w[1], b[0]-b[1])
# plot_line([0,4], w[0]-w[2], b[0]-b[2])
# plot_line([3.5, 8], w[1]-w[2], b[1]-b[2])
# processDataPerClassLogisticRegression(t, 2)

