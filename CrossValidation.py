import random

# U Cross Validaciji data set izaberemo na slucajan nacin
# Potom taj skup podataka podelimo u tri dela
# Trening skup - 60% od data set-a (3/5 od data set-a)
# Validacioni skup - 20% od data set-a (1/5 od data set-a)
# Test skup - 20% od data set-a (1/5 od data set-a)

def makeSets(data):
    data = random.sample(data, len(data))
    k = len(data) // 5
    lenTraingSet = 3 * k
    trainingSet = data[0:lenTraingSet]

    lenTestSet = k
    testSet = data[lenTraingSet:(lenTraingSet + lenTestSet)]

    lenValidSet = k
    validSet = data[(lenTraingSet + lenValidSet):]

    return [trainingSet, testSet, validSet]