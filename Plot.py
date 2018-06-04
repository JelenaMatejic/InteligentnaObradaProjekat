import matplotlib.markers as mark
import matplotlib.pyplot as plt
import random
import numpy as np

# Crtanje oznacenih podataka data set-a
def plotData(x, t):
    #izdvojimo sve moguce klase koje imamo (tj. izdvojimo sve razlicite labele)
    classes = np.unique(t)

    # izdvojimo sve moguce markere
    markers = []
    for m in mark.MarkerStyle.markers:
        markers.append(m)

    # idemo po klasama i iscrtavamo podatke - podaci jedne klase imaju nasumicno odabranu boju i oblik (marker)
    for i in classes:
        plt.scatter(x.T[0][t[0] == i], x.T[1][t[0] == i], marker=random.choice(markers))

def plotLine(x, w, b):
    xMax = max(x[:][:, 0])
    xMin = min(x[:][:,0])
    xCord = [xMin, xMax] # zelim da hiperravan bude dovoljno duga i deli podatke od prvog do poslenjeg postojeceg podatka
    yCord = [-(w[0, 0] / w[0, 1]) * x - (b[0, 0] / w[0, 1]) for x in xCord] # odredimo y koordinate za prethodno odredjene x koordinate
    plt.plot(xCord, yCord)

def plotLinesMulticlassOneVsAll(x, t, w, b):
    labels = np.unique(t)
    for i in range(len(labels)):
        plotLine(x, w[i], b[i])
