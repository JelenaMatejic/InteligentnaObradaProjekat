import matplotlib.pyplot as plt
import random
import numpy as np
from Plotting import Markers

def plotData(x, t):
    classes = np.unique(t) #izdvojimo sve moguce klase koje imamo (tj. izdvojimo sve razlicite labele)
    markers = Markers.listOfMarkers()
    # idemo po klasama i iscrtavamo podatke - podaci jedne klase imaju nasumicno odabranu boju i oblik (marker)
    for i in classes:
        m = random.choice(markers)
        plt.scatter(x.T[0][t[0] == i], x.T[1][t[0] == i], marker=m)

def lineCoordinates(x, w, b):
    xMax = max(x[:][:, 0])
    xMin = min(x[:][:, 0])
    xCord = [xMin-2,xMax+2]  # zelim da hiperravan bude dovoljno duga i deli podatke od prvog do poslenjeg postojeceg podatka
    yCord = [-(w[0, 0] / w[0, 1]) * x - (b[0, 0] / w[0, 1]) for x in xCord]  # odredimo y koordinate za prethodno odredjene x koordinate
    return [xCord, yCord]

def plotLine(x, w, b):
    xCord, yCord = lineCoordinates(x, w, b)
    plt.plot(xCord, yCord)



