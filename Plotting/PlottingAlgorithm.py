import CrossValidation
import Initializing
import ReadingFromFile
from Algorithms import Perceptron, LogisticRegression, PassiveAggressiveAlgorithm
from Plotting import Plot, PlotInWindow

def plotAlgorithmInWindow(file, algorithm):
    data = ReadingFromFile.readDataFromFile(file, ',')  # Podaci ucitani iz fajla
    trainingSet, testSet = CrossValidation.makeSets(data)
    x, t = Initializing.processData(trainingSet)
    t = Initializing.checkLabels(t, algorithm)
    kTrainingSets, kValidSets = CrossValidation.kCrossValidationMakeSets(trainingSet,5)  # Napravimo k trening i test set-ova unakrsnom validacijom (k = 5)

    if algorithm == "perceptron":
        w, b = Initializing.initialParam(x)
        w, b = Perceptron.train(x, t, w, b)
    elif algorithm == "logistic":
        w, b = LogisticRegression.crossTrain(kTrainingSets,kValidSets)  # Istreniramo k trening setova i kao rezultat vratimo najbolje w i najbolje b  (ono w i b za koje je greska bila najmanja)
    else:
        c = PassiveAggressiveAlgorithm.optC(kTrainingSets, kValidSets)  # Podesimo optimalni parametar c
        w, b = PassiveAggressiveAlgorithm.crossTrain(kTrainingSets, kValidSets, c)  # Istreniramo k trening setova i kao rezultat vratimo najbolje w i najbolje b  (ono w i b za koje je greska bila najmanja)

    xTest, tTest = Initializing.processData(testSet)
    fig = PlotInWindow.plotInWindow(xTest, w, b, tTest)
    return fig

