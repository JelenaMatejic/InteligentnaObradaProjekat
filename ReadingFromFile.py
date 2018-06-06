def readDataFromFile(fileName, splitChar):
    data = []
    fileName = fileName.replace("\\","/")
    textFile = open(fileName, "r")
    # Svaka od linija u .txt fajlu treba biti jedan pimer tj. jedan vektor u data set-u
    for line in textFile:
        line = line.strip('\n')
        sample = line.split(splitChar)
        sample = [float(i) for i in sample]
        data.append(sample)
    return data

