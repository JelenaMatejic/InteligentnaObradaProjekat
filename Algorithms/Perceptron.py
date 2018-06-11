import Initializing
from Plotting import PlotInWindow
import numpy as np

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

def predict(testSet, w, b):
    xTest, tTest = Initializing.processData(testSet)
    predictSet = np.dot(w,xTest.T) + b
    predictSet[predictSet > 0] = 1
    predictSet[predictSet <= 0] = -1
    return predictSet

def perceptronPlotInWindow(file):
    fig = PlotInWindow.plotAlgorithmInWindow(file, "perceptron")
    return fig