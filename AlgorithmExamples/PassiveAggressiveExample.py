import ReadingFromFile
import Initializing
import Plot
import CrossValidation
import Algorithms.PassiveAggressiveAlgorithm as PassiveAggressiveAlgorithm
import matplotlib.pyplot as plt

def PassiveAggressiveAlgorithmExample(file):
    file = ReadingFromFile.checkFile(file, "passiveAggressive")
    data = ReadingFromFile.readDataFromFile(file, ',')
    trainingSet, testSet = CrossValidation.makeSets(data)  # Napravimo trening i test set
    kTrainingSets, kValidSets = CrossValidation.kCrossValidationMakeSets(trainingSet,5)  # Napravimo k trening i test set-ova unakrsnom validacijom (k = 5)
    c = PassiveAggressiveAlgorithm.optC(kTrainingSets, kValidSets)
    w, b = PassiveAggressiveAlgorithm.crossTrain(kTrainingSets, kValidSets, c)  # Istreniramo k trening setova i kao rezultat vratimo najbolje w i najbolje b  (ono w i b za koje je greska bila najmanja)

    x, t = Initializing.processData(testSet)  # Rezultat crtamo i merimo nad test skupom podataka
    t = Initializing.checkLabels(t, "passiveAggressive")
    Plot.plotData(x, t)
    Plot.plotLine(x, w, b)
    plt.show()

PassiveAggressiveAlgorithmExample("../dataSets/PerceptronDataSet.txt")