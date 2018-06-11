import random

# Izmesamo pocetni skup podataka (data)
# 80% podataka ovog skupa nam je trening skup podataka
# 20% podataka ovog skupa je test skup podataka
# Od trening skupa podataka pravimo k skupva za treniranje i validaciju
# Podelimo data skup na k podskupova (delova) i idemo redom.
# U sakom prolazu tekuci podskup predstavlja test skup podataka, a preostali podskupovi cine trening skup podataka

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
