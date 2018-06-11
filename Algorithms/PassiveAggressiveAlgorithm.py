import ReadingFromFile
import CrossValidation
import Initializing
import Plot
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

def train(x, t, w, b, c, epohe = 15):
    for e in range(epohe):
        for n in range(len(x)):
            tn = np.array(t[0,n]).reshape((1, 1))
            xn = np.array(x[n]).reshape((1,len(x[n])))
            ln = loss(tn, w, xn)
            lamN = lambd(ln, xn, c)

            # pomerimo w i b
            if ln > 0:
                w = w + np.dot(np.dot(lamN,tn), xn)
                b = b + np.dot(lamN, tn)

    return [w, b]

def sumLoss(xValid, tValid, w):
    sum = 0
    for n in range(len(xValid)):
        tn = np.array(tValid[0,n]).reshape((1, 1))
        xn = np.array(xValid[n]).reshape((1, len(xValid[n])))
        ln = loss(tn, w, xn)
        sum += ln
    return sum

def optC(kTrainingSets, kValidSets, cSet = [0.001, 0.01, 0.1, 10, 100, 1000]):
    minLoss = math.inf # posto su gubici negativni, najmanji je onajkoji je najblizi nuli
    optimalC = cSet[0]
    for c in cSet:
        w, b = crossTrain(kTrainingSets, kValidSets, c)
        xValid, tValid = Initializing.processData(kValidSets[0])
        sumL = sumLoss(xValid, tValid, w)
        if minLoss > abs(sumL[0][0]):
            minLoss = abs(sumL[0][0])
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

def crossTrain(kTrainingSets, kValidSets, c):
    minLoss = math.inf
    bestW = math.inf
    bestB = math.inf
    for k in range(len(kTrainingSets)):
        # Trenirano parametre na k-tom trening set-u
        x, t = Initializing.processData(kTrainingSets[k])
        xValid, tValid = Initializing.processData(kValidSets[k])
        w, b = Initializing.initialParam(x)
        w, b = train(x, t, w, b, c)
        kLoss = sumLoss(xValid, tValid, w) # Merino gresku na k-tom validacionom set-u, za dobijene parametre w i t
        if minLoss > abs(kLoss[0][0]):
            minLoss = abs(kLoss[0][0])
            bestW = w
            bestB = b

    return [bestW, bestB]

def passiveAggressivePlotInWindow(file):
    data = ReadingFromFile.readDataFromFile(file, ',')  # Podaci ucitani iz fajla
    trainingSet, testSet = CrossValidation.makeSets(data)  # Napravimo trening i test set
    kTrainingSets, kValidSets = CrossValidation.kCrossValidationMakeSets(trainingSet, 5)  # Napravimo k trening i test set-ova unakrsnom validacijom (k = 5)

    c = optC(kTrainingSets, kValidSets) # Podesimo optimalni parametar c
    w, b = crossTrain(kTrainingSets, kValidSets, c)  # Istreniramo k trening setova i kao rezultat vratimo najbolje w i najbolje b  (ono w i b za koje je greska bila najmanja)
    x, t = Initializing.processData(testSet)  # Rezultat crtamo i merimo nad test skupom podataka
    t = Initializing.checkLabels(t, "passiveAggressive")

    fig = Plot.plotInWindow(x, w, b, t)
    return fig


