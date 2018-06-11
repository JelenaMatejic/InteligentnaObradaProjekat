import matplotlib.pyplot as plt

import Algorithms.LogisticRegression as LogisticRegression
import Initializing
import ReadingFromFile
from Algorithms import CrossValidation
from Plotting import Plot


def logisticRegressionExample(file):
    file = ReadingFromFile.checkFile(file, "logistic")
    data = ReadingFromFile.readDataFromFile(file, ',')  # Podaci ucitani iz fajla
    trainingSet, testSet = CrossValidation.makeSets(data)  # Napravimo trening i test set
    kTrainingSets, kValidSets = CrossValidation.kCrossValidationMakeSets(trainingSet, 5)  # Napravimo k trening i test set-ova unakrsnom validacijom (k = 5)

    w, b = LogisticRegression.crossTrain(kTrainingSets, kValidSets)  # Istreniramo k trening setova i kao rezultat vratimo najbolje w i najbolje b  (ono w i b za koje je greska bila najmanja)
    x, t = Initializing.processData(testSet)  # Rezultat crtamo i merimo nad test skupom podataka
    t = Initializing.checkLabels(t, "logistic")
    Plot.plotData(x, t)
    Plot.plotLine(x, w, b)
    plt.show()

logisticRegressionExample('../dataSets/LogisticRegressionDataSet.txt')