import Initializing
import Plot
import LogisticRegression
import Perceptron
import PassiveAggressiveAlgorithm
import ReadingFromFile
import CrossValidation
import numpy as np
import matplotlib.markers as mark
import matplotlib.pyplot as plt
import random
import copy
import math

def processDataPerClass(t, classId, algorithm):
    if algorithm == 'logistic':
        negClass = 0
    else:
        negClass = -1
    tmpT = copy.copy(t)
    # podaci cije su labele jednake prosledjenoj klasi u logistickoj regresiji dobijaju vrednost 1, u suprotnom 0
    for tn in range(len(t[0])):
        if t[0][tn] == classId:
            tmpT[0][tn] = 1
        else:
            tmpT[0][tn] = negClass
    return tmpT

def processDataSetPerClass(data, classId, algorithm):
    tmpData = copy.deepcopy(data)
    if algorithm == "logistic":
        negativeClass = 0
    else:
        negativeClass = -1

    for i in range(len(tmpData)):
        if data[i][-1] == classId:
            tmpData[i][-1] = 1
        else:
            tmpData[i][-1] = negativeClass
    return tmpData

def trainClassifiers(data, algorithm):
    #data = [[1,1,0], [1,2,0], [5,5,1], [4,5,1], [-5,-4,2], [-5,-5,2]]
    print(data)
    print()
    x, t = Initializing.processData(data)
    labels = np.unique(t)
    listW = []
    listB = []
    for i in range(len(labels)):
        tmpData = processDataSetPerClass(data, i, algorithm)
        tmpX, tmpT = Initializing.processData(tmpData)
        w, b = Initializing.initialParam(tmpX)
        if algorithm == 'perceptron':
            w, b = Perceptron.train(tmpX, tmpT, w, b)
        elif algorithm == 'logistic':
            # w, b = LogisticRegression.train(tmpX, tmpT, w, b)
            LogisticRegression.bestW, LogisticRegression.bestB = Initializing.initialParam(tmpX)
            print(LogisticRegression.bestW, LogisticRegression.bestB)
            trainingSet, testSet = CrossValidation.makeSets(tmpData)  # Napravimo trening i test set
            kTrainingSets, kValidSets = CrossValidation.kCrossValidationMakeSets(trainingSet, 5)  # Napravimo k trening i test set-ova unakrsnom validacijom (k = 5)
            w, b = LogisticRegression.crossTrain(kTrainingSets,kValidSets)  # Istreniramo k trening setova i kao rezultat vratimo najbolje w i najbolje b  (ono w i b za koje je greska bila najmanja)
        else:
            trainingSet, testSet = CrossValidation.makeSets(tmpData)  # Napravimo trening i test set
            kTrainingSets, kValidSets = CrossValidation.kCrossValidationMakeSets(trainingSet,5)  # Napravimo k trening i test set-ova unakrsnom validacijom (k = 5)
            c = PassiveAggressiveAlgorithm.optC(kTrainingSets, kValidSets)  # Podesimo optimalni parametar c
            w, b = PassiveAggressiveAlgorithm.crossTrain(kTrainingSets, kValidSets, c)  # Istreniramo k trening setova i kao rezultat vratimo najbolje w i najbolje b  (ono w i b za koje je greska bila najmanja)
        listW.append(np.array(w[0]).reshape(1,len(w[0])))
        listB.append(np.array(b[0]).reshape(1,1))
        print(w, b)
        print(LogisticRegression.bestW, LogisticRegression.bestB)
        print()
    return [listW, listB]

def oneVsAllPerceptronExample(file):
    data = ReadingFromFile.readDataFromFile(file, ',')

    x, t = Initializing.processData(data)  # iz pocetnog skupa podataka razdvojimo podatke i labele
    listW, listB = trainClassifiers(data, "perceptron")

    fig = Plot.plotLinesMulticlassOneVsAll(x, t, listW, listB)
    return fig

def oneVsAllLogisticRegressionExample(file):
    data = ReadingFromFile.readDataFromFile(file, ',')

    x, t = Initializing.processData(data)  # iz pocetnog skupa podataka razdvojimo podatke i labele
    listW, listB = trainClassifiers(data, "logistic")

    fig = Plot.plotLinesMulticlassOneVsAll(x, t, listW, listB)
    return fig

def oneVsAllPassiveAggressiveExample(file):
    data = ReadingFromFile.readDataFromFile(file, ',')

    x, t = Initializing.processData(data)  # iz pocetnog skupa podataka razdvojimo podatke i labele
    listW, listB = trainClassifiers(data, "passiveAgressive")

    fig = Plot.plotLinesMulticlassOneVsAll(x, t, listW, listB)
    return fig

#oneVsAllLogisticRegressionExample()
#oneVsAllPerceptronExample()
#oneVsAllPerceptronExample()


