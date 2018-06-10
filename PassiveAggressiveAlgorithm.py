import ReadingFromFile
import CrossValidation
import Initializing
import Plot
import random
import math
import numpy as np
import matplotlib.pyplot as plt

def loss(tn, w, xn):
    tmp = np.dot(w, xn.T)
    ln = 1 - np.dot(tn, tmp)
    return ln

def lambd(ln, xn, c):
    xn2 = np.dot(xn, xn.T)
    tmp = ln[0]/xn2
    l = min(c, tmp)
    return l

def train(x, t, w, b, c = 0.01, epohe = 100):
    for e in range(epohe):
        for n in range(len(x)):
            tn = np.array(t[0,n])
            xn = np.array(x[n])
            ln = loss(tn, w, xn)
            lamN = lambd(ln, xn, c)

            # pomerimo w i b
            if ln > 0:
                w = w + np.dot(lamN*tn, xn)
                b = b + np.dot(lamN, tn)

    return [w, b]

def sumLoss(trainingSet, w, b, c):
    sum = 0
    #t = labels(trainingSet)
    for n in range(len(trainingSet)):
        tn = np.array(t[n]).reshape((1, 1))
        xn = np.array(trainingSet[n][:-1]).reshape((1, len(trainingSet[n]) - 1))
        ln = loss(tn, w, xn)
        sum += ln

    return sum

def optC(trainingSet, validSet, w, b, cSet = [0.01, 0.1, 10, 100, 1000]):
    minLoss = math.inf
    optimalC = cSet[0]
    for c in cSet:
        w = np.zeros((1, len(data[0]) - 1))
        b = np.array([[0]])
        w, b = train(trainingSet, w, b, c)
        sumL = sumLoss(validSet, w, b, c)
        if minLoss > sumL:
            minLoss = sumL
            optimalC = c
    return [optimalC]

def testResults(testSet, w, b, c):
    t = labels(testSet)
    tResult = []
    for n in range(len(testSet)):
        tn = np.array(t[n]).reshape((1, 1))
        xn = np.array(testSet[n][:-1]).reshape((1, len(testSet[n]) - 1))
        if (np.dot(w, xn.T) + b) > 0:
            tResult.append(1)
        else:
            tResult.append(-1)
    return tResult

def crossTrain(kTrainingSets, kValidSets):

    for k in range(len(kTrainingSets)):
        # Trenirano parametre na k-tom trening set-u
        x, t = Initializing.processData(kTrainingSets[k])
        w, b = Initializing.initialParam(x)
        w, b = train(x, t, w, b)

        # # Merino gresku na k-tom validacionom set-u, za dobijene parametre w i t
        # xValid, tValid = Initializing.processData(kValidSets[k])
        # kLoss = cost(xValid, tValid, w, b)
        # #print(kLoss)
        # minCost(w, b, kLoss)

    # Nakon unakrsnog treninga i validacije, vratimo najbolje parametra w i b, nad kojima cemo testirati
    # global bestW, bestB
    # return [bestW, bestB]
    return [w, b]

data = ReadingFromFile.readDataFromFile('./dataSets/PerceptronDataSet.txt', ',')
trainingSet, testSet = CrossValidation.makeSets(data)  # Napravimo trening i test set
kTrainingSets, kValidSets = CrossValidation.kCrossValidationMakeSets(trainingSet,5)  # Napravimo k trening i test set-ova unakrsnom validacijom (k = 5)
w, b = crossTrain(kTrainingSets, kValidSets)  # Istreniramo k trening setova i kao rezultat vratimo najbolje w i najbolje b  (ono w i b za koje je greska bila najmanja)

x, t = Initializing.processData(testSet)  # Rezultat crtamo i merimo nad test skupom podataka
Plot.plotData(x, t)
Plot.plotLine(x, w, b)
plt.show()
