import Algorithms.Perceptron as Perceptron
import ReadingFromFile
import CrossValidation
import Initializing
import Plot
import matplotlib.pyplot as plt

def perceptronExample(file):
    file = ReadingFromFile.checkFile(file, "perceptron")
    data = ReadingFromFile.readDataFromFile(file, ',') # Podaci ucitani iz fajla
    trainingSet, testSet = CrossValidation.makeSets(data)

    x, t = Initializing.processData(trainingSet)
    t = Initializing.checkLabels(t, "perceptron")
    w, b = Initializing.initialParam(x)
    w, b = Perceptron.train(x, t, w, b)
    Plot.plotData(x, t)
    Plot.plotLine(x, w, b)
    plt.show()

perceptronExample("../dataSets/PerceptronDataSet.txt")