import Initializing
from Plotting import PlottingAlgorithm
import numpy as np
import math

bestW = math.inf
bestB = math.inf
minLoss = math.inf

def sigmoid(z):
    sigm = 1/(1 + np.exp(-z))
    return sigm

def output(x, w, b):
    a = np.dot(w,x.T)+b
    return sigmoid(a)

def train(x, t, w, b, epoha = 100, learnRate = 0.1):
    for e in range(epoha):
        for n in range(len(x)):
            w = w - learnRate * (output(x[n], w, b) - t[0,n]) * x[n].T
            b = b - learnRate * (output(x[n], w, b) - t[0,n])
    return [w,b]

def cost(x, t, w, b):
    loss = t * np.log(output(x, w, b)) + (1 - t) * np.log(1-output(x, w, b))
    return np.sum(loss)

def minCost(w, b, loss):
    global minLoss, bestW, bestB
    if minLoss > loss:
        minLoss = loss
        bestW = w
        bestB = b

def predict(new_x, w, b):
    out = output(new_x, w, b)
    out[out > 0.5] = 1
    out[out < 0.5] = 0
    return out

def crossTrain(kTrainingSets, kValidSets):
    global bestW, bestB, minLoss
    bestW = math.inf
    bestB = math.inf
    minLoss = math.inf
    for k in range(len(kTrainingSets)):
        # Trenirano parametre na k-tom trening set-u
        x, t = Initializing.processData(kTrainingSets[k])
        w, b = Initializing.initialParam(x)
        w, b = train(x, t, w, b)

        # Merino gresku na k-tom validacionom set-u, za dobijene parametre w i t
        xValid, tValid = Initializing.processData(kValidSets[k])
        kLoss = cost(xValid, tValid, w, b)
        minCost(w, b, kLoss)

    # Nakon unakrsnog treninga i validacije, vratimo najbolje parametra w i b, nad kojima cemo testirati
    return [bestW, bestB]

def logisticRegressionPlotInWindow(file):
    fig = PlottingAlgorithm.plotAlgorithmInWindow(file, "logistic")
    return fig

