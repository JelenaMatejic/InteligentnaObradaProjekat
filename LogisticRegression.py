import Initializing
import Plot
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    sigm = 1/(1 + np.exp(-z))
    return sigm

def output(x, w, b):
    a = np.dot(w,x.T)+b
    return sigmoid(a)

def trein(w, b, x, t, epoha = 1000, learnRate = 0.1):
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
    data = [[4, 2, 1], [1, 1, 0], [2, 2, 0], [2, 3, 0], [3, 2, 0], [5, 1, 1], [5, 2, 1], [6, 2, 1], [6, 3, 1],
            [6, 6, 1]]
    x, t = Initializing.processData(data) # izdvajanje x i t iz skupa data
    w, b = Initializing.initialParam(x) # pocetne vrednosti za w i b
    Plot.plotData(x, t) # crtanje podataka iz skupa data
    w, b = trein(w, b, x, t) # treniranje modela
    Plot.plotLine(x, w, b) # crtanje hiperravni
    plt.show()

#logisticRegressionExample()
