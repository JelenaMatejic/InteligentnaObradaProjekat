import numpy as np
from Plotting import Plot
from matplotlib.figure import Figure
from Plotting import Markers
import matplotlib.pyplot as plt
import random

fig = Figure()
a = fig.add_subplot(111)

def plotLinesMulticlassOneVsAll(x, t, listW, listB):
    labels = np.unique(t)
    for i in range(len(labels)):
        plotLineInWindow(x, listW[i], listB[i])
    global a, fig
    plotDataInWindow(x, t, a, fig)
    return fig

def plotInWindow(x, w, b, t):
    global a, fig
    a, fig = plotLineInWindow(x, w, b)
    plotDataInWindow(x, t, a, fig)
    return fig

def plotLineInWindow(x, w, b):
    xCord, yCord = Plot.lineCoordinates(x, w, b)

    global a, fig
    a = fig.add_subplot(111)
    a.plot(xCord, yCord)
    return [a, fig]

def plotDataInWindow(x, t, a, fig):
    #izdvojimo sve moguce klase koje imamo (tj. izdvojimo sve razlicite labele)
    classes = np.unique(t)
    markers = Markers.listOfMarkers()

    plt.draw()
    # idemo po klasama i iscrtavamo podatke - podaci jedne klase imaju nasumicno odabranu boju i oblik (marker)
    for i in classes:
        m = random.choice(markers)
        a.scatter(x.T[0][t[0] == i], x.T[1][t[0] == i], marker=m)
    return fig
