import numpy as np

def processData(data):
    data = np.array(data)

    # izdvojimo labele t = [1 x N]
    t = data[:, -1]
    t = np.array(t).reshape((1, len(data)))

    # izdvojimo podatke x = [N x 2]
    x = data[:, :-1]
    x = np.array(x).reshape((len(data), len(data[0]) - 1))

    return [x,t]

def initialParam(x):
    w = np.zeros((1, len(x[0])))  # inicijalno w je nula vektor dimenzija w = [1 x M]
    b = np.array([[0]]) # inicijalno b je nula vektor dimenzija b = [1 x 1]
    return [w,b]

def checkLabels(t, algorithm):
    if algorithm == "logistic":
        negativeClass = 0
    else:
        negativeClass = -1

    for i in range(len(t[0])):
        if t[0][i] != 1:
            t[0][i] = negativeClass
    return t
