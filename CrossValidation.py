import random
import ReadingFromFile

# U Cross Validaciji data set izaberemo na slucajan nacin
# Potom taj skup podataka podelimo u tri dela
# Trening skup - 60% od data set-a (3/5 od data set-a)
# Validacioni skup - 20% od data set-a (1/5 od data set-a)
# Test skup - 20% od data set-a (1/5 od data set-a)

def makeSets(data):
    data = random.sample(data, len(data))
    k = len(data) // 5
    lenTraingSet = 4 * k
    trainingSet = data[0:lenTraingSet]
    testSet = data[lenTraingSet:]

    return [trainingSet, testSet]

def kCrossValidationMakeSets(trainingSet, k):
    setLen = len(trainingSet) // k;
    kTrainingSets = []
    kValidSets = []
    for i in range(k):
        train = []
        startInd = i*setLen
        endInd = (i+1)*setLen
        kValidSets.append(trainingSet[startInd:endInd])
        if startInd == 0:
            train += trainingSet[endInd:]
        else:
            train += trainingSet[0:startInd]
            train += trainingSet[endInd:]
        kTrainingSets.append(train)

    return [kTrainingSets, kValidSets]

data = ReadingFromFile.readDataFromFile('PerceptronDataSet.txt', ',')
kCrossValidationMakeSets(data, 3)