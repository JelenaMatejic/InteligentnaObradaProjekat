import matplotlib.markers as mark
import matplotlib.pyplot as plt
import random
import numpy as np
from matplotlib.figure import Figure

def listOfMarkers():
    # izdvojimo sve moguce markere
    markers = []
    for m in mark.MarkerStyle.markers:
        markers.append(m)
    return markers

def plotData(x, t):
    #izdvojimo sve moguce klase koje imamo (tj. izdvojimo sve razlicite labele)
    classes = np.unique(t)
    markers = listOfMarkers()
    # idemo po klasama i iscrtavamo podatke - podaci jedne klase imaju nasumicno odabranu boju i oblik (marker)
    for i in classes:
        plt.scatter(x.T[0][t[0] == i], x.T[1][t[0] == i], marker=random.choice(markers))

def lineCoordinates(x, w, b):
    xMax = max(x[:][:, 0])
    xMin = min(x[:][:, 0])
    xCord = [xMin,xMax]  # zelim da hiperravan bude dovoljno duga i deli podatke od prvog do poslenjeg postojeceg podatka
    yCord = [-(w[0, 0] / w[0, 1]) * x - (b[0, 0] / w[0, 1]) for x in xCord]  # odredimo y koordinate za prethodno odredjene x koordinate
    return [xCord, yCord]

def plotLine(x, w, b):
    xCord, yCord = lineCoordinates(x, w, b)
    plt.plot(xCord, yCord)

def plotLinesMulticlassOneVsAll(x, t, w, b):
    labels = np.unique(t)
    for i in range(len(labels)):
        plotLine(x, w[i], b[i])

def plotInWindow(x, w, b, t):
    a, fig = plotLineInWindow(x, w, b)
    plotDataInWindow(x, t, a, fig)
    return fig

def plotLineInWindow(x, w, b):
    xCord, yCord = lineCoordinates(x, w, b)

    dim = xCord[1] - xCord[0] + 1
    fig = Figure(figsize=(dim, dim))
    a = fig.add_subplot(111)
    a.plot(xCord, yCord)
    return [a, fig]

def plotDataInWindow(x, t, a, fig):
    #izdvojimo sve moguce klase koje imamo (tj. izdvojimo sve razlicite labele)
    classes = np.unique(t)
    markers = listOfMarkers()
    # idemo po klasama i iscrtavamo podatke - podaci jedne klase imaju nasumicno odabranu boju i oblik (marker)
    for i in classes:
        a.scatter(x.T[0][t[0] == i], x.T[1][t[0] == i], marker=random.choice(markers))
    return fig



