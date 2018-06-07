import Initializing
import Plot
import ReadingFromFile
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    sigm = 1/(1 + np.exp(-z))
    return sigm

def output(x, w, b):
    a = np.dot(w,x.T)+b
    return sigmoid(a)

def trein(x, t, w, b, epoha = 1000, learnRate = 0.1):
    for e in range(epoha):
        for n in range(len(x)):
            w = w - learnRate * (output(x[n], w, b) - t[0,n]) * x[n].T
            b = b - learnRate * (output(x[n], w, b) - t[0,n])
    return [w,b]

def cost(x, t, w, b):
    loss = t * np.log(output(x, w, b)) + (1 - t) * np.log(1-output(x, w, b))
    return np.sum(loss)

def predict(new_x, w, b):
    out = output(new_x, w, b)
    out[out > 0.5] = 1
    out[out < 0.5] = 0
    return out

def logisticRegressionExample():
    data = ReadingFromFile.readDataFromFile("LogisticRegressionDataSet.txt", ',')
    x, t = Initializing.processData(data) # izdvajanje x i t iz skupa data
    w, b = Initializing.initialParam(x) # pocetne vrednosti za w i b
    Plot.plotData(x, t) # crtanje podataka iz skupa data
    w, b = trein(x, t, w, b) # treniranje modela
    Plot.plotLine(x, w, b) # crtanje hiperravni
    plt.show()

# def logisticRegressionPlotInWindow():
#     data = ReadingFromFile.readDataFromFile("LogisticRegressionDataSet.txt", ',')
#     trainingSet, testSet, validSet = CrossValidation.makeSets(data)
#     x, t = Initializing.processData(trainingSet)
#     w, b = Initializing.initialParam(x)
#     w, b = train(x, t, w, b)
#     fig = Plot.plotInWindow(x, w, b, t)
#     return fig

logisticRegressionExample()