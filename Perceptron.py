import CrossValidation
import ReadingFromFile
import Initializing
import Plot
import matplotlib.pyplot as plt
import numpy as np
import math

minLoss = math.inf
bestW = math.inf
bestB = math.inf

# Treniranje modela
def train(x, t, w, b, epoha = 100, learnRate = 1):
    for e in range(epoha):
        for n in range(len(x)):
            wx = np.dot(w, x[n].T) + b
            tn = t[0][n]
            if wx * tn <= 1: # ako je (wx+b)*t <= 1 tada je primer lose klasifikovan i pomerimo w i b
                w = w + x[n]*tn
                b = b + tn
        if cost(x, t, w, b) == 0: # ako je gubitak = 0 onda je pronadjena hiperravan koja dobro deli podatke
            #print("Konvergencija postignuta u epohi: " + str(e))
            break
    return [w,b]

# Funkcija gubitka za perceptron je max(0, -w*x*t)
def cost(x, t, w, b):
    loss = (np.dot(w,x.T) + b) * t # gubitak na celom skupu
    loss[loss > 0] = 0 # ako je gubitak na n-tom primeru > 0 onda je loss = 0
    loss = (-1)*loss # pomnozimo sa -1 da bi negativni gubici postali pozitivni
    return np.sum(loss)

def minCost(w, b, loss):
    global minLoss
    global bestW
    global bestB
    if minLoss > loss:
        minLoss = loss
        bestW = w
        bestB = b

def predict(testSet, w, b):
    predictSet = np.dot(w,testSet.T) + b
    predictSet[predictSet > 0] = 1
    predictSet[predictSet <= 0] = -1
    return predictSet

def crossTrain(kTrainingSets, kValidSets):

    for k in range(len(kTrainingSets)):
        # Trenirano parametre na k-tom trening set-u
        x, t = Initializing.processData(kTrainingSets[k])
        w, b = Initializing.initialParam(x)
        w, b = train(x, t, w, b)

        # Merino gresku na k-tom validacionom set-u, za dobijene parametre w i t
        xValid, tValid = Initializing.processData(kValidSets[k])
        kLoss = cost(xValid, tValid, w, b)
        minCost(w, b, kLoss)

    # Nakon unakrsnog treninga i testiranja, vratimo najbolje parametra w i b, nad kojima cemo testirati
    global bestW, bestB
    return [bestW, bestB]

def perceptronExample():
    data = ReadingFromFile.readDataFromFile('PerceptronDataSet.txt', ',') # Podaci ucitani iz fajla
    trainingSet, testSet = CrossValidation.makeSets(data) # Napravimo trening i test set
    kTrainingSets, kValidSets = CrossValidation.kCrossValidationMakeSets(trainingSet, 5) # Napravimo k trening i test set-ova unakrsnom validacijom (k = 5)

    w, b = crossTrain(kTrainingSets, kValidSets) # Istreniramo k trening setova i kao rezultat vratimo najbolje w i najbolje b  (ono w i b za koje je greska bila najmanja)
    x, t = Initializing.processData(testSet) # Rezultat crtamo i merimo nad test skupom podataka
    Plot.plotData(x, t)
    Plot.plotLine(x, w, b)
    plt.show()

def perceptronPlotInWindow():
    data = ReadingFromFile.readDataFromFile('PerceptronDataSet.txt', ',')  # Podaci ucitani iz fajla
    trainingSet, testSet = CrossValidation.makeSets(data)  # Napravimo trening i test set
    kTrainingSets, kValidSets = CrossValidation.kCrossValidationMakeSets(trainingSet, 5)  # Napravimo k trening i test set-ova unakrsnom validacijom (k = 5)

    w, b = crossTrain(kTrainingSets, kValidSets)  # Istreniramo k trening setova i kao rezultat vratimo najbolje w i najbolje b  (ono w i b za koje je greska bila najmanja)
    x, t = Initializing.processData(testSet)  # Rezultat crtamo i merimo nad test skupom podataka

    fig = Plot.plotInWindow(x, w, b, t)
    return fig

#perceptronExample()