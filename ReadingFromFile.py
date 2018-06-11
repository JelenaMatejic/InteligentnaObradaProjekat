def readDataFromFile(fileName, splitChar):
    data = []
    textFile = open(fileName, "r")
    # Svaka od linija u .txt fajlu treba biti jedan pimer tj. jedan vektor u data set-u
    for line in textFile:
        line = line.strip('\n')
        sample = line.split(splitChar)
        sample = [float(i) for i in sample]
        data.append(sample)
    return data

def checkFile(file, algorithm):
    if file == "" or file == "No file chosen . . .":
        if algorithm == "perceptron":
            file = "../dataSets/PerceptronDataSet.txt"
        elif algorithm == "logistic":
            file = "../dataSets/LogisticRegressionDataSet.txt"
        else:
            filr = "./dataSets/PerceptronDataSet.txt"
    return file

