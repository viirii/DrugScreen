import random

import numpy as np
import math
from sklearn.metrics import f1_score

from reader import read_train_test, read_blind, write_prediction
import sys

Initial = 500
LIMIT = 2500
CAP = 2500

# update the weights after comparing with label
def updateWeights(answer, weights, data, i):
    newWeights = [1 for x in range(len(data[0]))]
    for j in range(len(weights)):
        if answer == 0 and data[i][j] == 1:
            newWeights[j] = weights[j] * 0.5
        elif answer == 1 and data[i][j] == 0:
            newWeights[j] = weights[j] * 2
        else:
            newWeights[j] = weights[j]
    return newWeights


def getPrediction(score, theta):
    for i in range(len(theta)):
        if score < theta[i]:
            return i
    return len(theta) - 1

def getRandomIndex(length, seen) :
    while True :
        randomInteger = random.randint(0, length-1)
        if randomInteger not in seen : return randomInteger

def getClosestPoint(weights, theta, data, seenIndex) :
    bestPoint = 0
    bestDist = math.inf
    for i in range(len(data)) :
        if i in seenIndex : continue
        else :
            #modifiedEuclideanDistance = sum(np.dot((data[i] - boundary), weights))
            modifiedEuclideanDistance = np.dot(data[i], weights) - theta[0]

            if modifiedEuclideanDistance < bestDist :
                bestPoint = i
                bestDist = modifiedEuclideanDistance
    return bestPoint, bestDist



def trainWeights(data, labels):
    # Initial Training

    weights = np.array([1 for x in range(len(data[0]))])
    numTheta = 1  # number of boundaries
    theta = [((i + 1) * len(data[0])) / numTheta for i in range(numTheta)]  # this is the decision threshold.
    errors = 0
    predictions = []
    seenIndex = []

    for i in range(Initial) : #range(len(data)):
        pickedIndex = getRandomIndex(len(labels), seenIndex)
        seenIndex.append(pickedIndex)
        xi = data[pickedIndex]
        score = np.dot(xi, weights)
        prediction = getPrediction(score, theta)
        predictions.append(prediction)
        correctAnswer = labels[pickedIndex]

        if correctAnswer != prediction:
            errors += 1
            weights = updateWeights(correctAnswer, weights, data, i)

    print("initial training done")

    # SVM part

    boundary = np.array([1/2 for x in range(len(data[0]))])

    for i in range(Initial, LIMIT):
        print(i)
        nextIndex, dist = getClosestPoint(weights, theta, data, seenIndex)
        seenIndex.append(nextIndex)
        xi = data[nextIndex]
        score = np.dot(xi, weights)
        prediction = getPrediction(score, theta)
        predictions.append(prediction)
        correctAnswer = labels[pickedIndex]

        if correctAnswer != prediction:
            errors += 1
            weights = updateWeights(correctAnswer, weights, data, i)

    print("*****" + str(len(predictions)))

    return weights, predictions, errors


def testWinnow(data, labels, weights) :
    errors = 0
    results = []
    for i in range(len(data)) :
        score = np.dot(data[i], weights)
        if score != labels[i] : errors += 1
        results.append(score)

    return errors, results


def winnow(difficulty='EASY') :
    X_train, y_train = read_train_test('{}_TRAIN.csv'.format(difficulty.upper()))
    X_test, y_test = read_train_test('{}_TEST.csv'.format(difficulty.upper()))
    weights, trainguess, trainerror = trainWeights(X_train, y_train)
    print(set(trainguess))

    print("ERROR RATE, TRAIN : " + str(trainerror/len(y_train)))

    svm_f1_score_tr = f1_score(y_train[:LIMIT], trainguess)
    print("F1 : " + str(svm_f1_score_tr))


    testerror, prediction = testWinnow(X_test, y_test, weights)
    print(set(prediction))
    svm_f1_score_tst = f1_score(y_test, prediction)
    print("F1 : " + str(svm_f1_score_tst))
    print("ERROR RATE, TEST : " + str(testerror/len(y_test)))





if __name__ == '__main__':
    if len(sys.argv) == 1:
        winnow(difficulty='EASY')
    else: winnow(difficulty=sys.argv[1])
