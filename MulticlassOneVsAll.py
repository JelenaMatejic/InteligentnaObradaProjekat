import Initializing
import Plot
import LogisticRegression
import Perceptron
import ReadingFromFile
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
            w, b = LogisticRegression.train(x, tmpT, w, b)
        listW.append(np.array(w[0]).reshape(1,len(w[0])))
        listB.append(np.array(b[0]).reshape(1,1))
    return [listW, listB]

def oneVsAllPerceptronExample(file):
    data = ReadingFromFile.readDataFromFile(file, ',')

    x, t = Initializing.processData(data)  # iz pocetnog skupa podataka razdvojimo podatke i labele
    listW, listB = trainClassifiers(x, t, "perceptron")

    fig = Plot.plotLinesMulticlassOneVsAll(x, t, listW, listB)
    return fig

def oneVsAllLogisticRegressionExample(file):
    data = ReadingFromFile.readDataFromFile('./dataSets/MulticlassOneVsAllDataSet.txt', ',')

    x, t = Initializing.processData(data)  # iz pocetnog skupa podataka razdvojimo podatke i labele
    listW, listB = trainClassifiers(x, t, "logistic")

    fig = Plot.plotLinesMulticlassOneVsAll(x, t, listW, listB)
    return fig

#oneVsAllLogisticRegressionExample()
#oneVsAllPerceptronExample()


