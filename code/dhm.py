import numpy as np
import random
import math

difficulty = 'EASY'


## This function runs the DHM and random learner in parallel assuming a streaming data model
# Input:  difficulty - the difficulty as a string, 'EAST' or "MODERATE'

# Additionally, you will implement a random learner for performing the
# same task and compare the performance of both algorithms

## ALGORITHM PARAMETERS
numsamples = 4000
# vectors for identifying points in sets S and T
SMask = np.array([0 for x in range(numsamples)])
TMask = np.array([0 for x in range(numsamples)])

# Labels for the points in S and T
Slabels = np.array([0 for x in range(numsamples)])
Tlabels = np.array([0 for x in range(numsamples)])

# R is a bit vector indicating which samples have been queried by a random learner
R = np.array([0 for x in range(numsamples)])
# generate the data.
#   XTrain is a 1 by numsamples vector of values in the interval [0,1].
#   YTrain is a 1 by numsamples vector of labels (either 0 or 1)
#   YTrain is the true model. You can use this to compute generalization error because abs(h-YTrain) = generalization error if h is the current model
XTrain, YTrain, XTest, YTest = readTrainTest(difficulty)


## select a random unlabeled point
def selectRandomUnlabeledPoint(r) :
    UR = np.where(r == 0)[0]  # get array of index of unlabeled points (random learning)
    xr = UR[random.randint(1, len(UR))-1]
    return xr

def sqrt(num) :
    return num**0.5

# *************** IMPLEMENT THIS   ***************** #
# You may need to create local variables to keep track of sets S and T, etc
cost = 0
# this is the main loop of the DHM algorithm
for t in range(numsamples) :

    # XTrain(t) is the next instance in the data stream

    # *************** IMPLEMENT THIS   ***************** #
    # you will need to:
    #   (i) learn the appropriate models by calling subroutineSVM
    #   (ii) apply the logic of the DHM algorithm
    #   (iii) append to DHMGeneralizationError after each call to the
    #   oracle.  i.e., DHMGeneralizationError(end+1)=abs(h-YTrain),
    #   where h is the current model, according to DHM
    #   (iv) implement a random learner that selects a *RANDOM* point each
    #   time DHM selects one.
    #   (v) append to RandGeneralizationError after each call to the
    #   oracle.  i.e., RandGeneralizationError(end+1)=abs(hr-YTrain),
    #   where hr is the current model, according to the random learner


    # Note that the DHM algorithm requires the calculation of Delta, the
    # generalization bound. The following code computes Delta. You should
    # use this (after computing hpluserr (the error by the h-plus-one
    # model) and hminuserr (the error by the h-minus-one-model). Of course,
    # you need to re-compute hpluserr and hminuserr each iteration.
    xr = t
    hneg, hnflag = subroutineSVM([XTrain(SMask == 1)XTrain(xr)], XTrain(TMask == 1), [Slabels(SMask == 1)0],
                                 Tlabels(TMask == 1))
    if hnflag == 1 :
        SMask[xr] = 1
        Slabels[xr] = 1
        continue

    hplus, hpflag = subroutineSVM([XTrain(SMask == 1) XTrain(xr)], XTrain(TMask == 1), [Slabels(SMask == 1)1],
                                  Tlabels(TMask == 1))
    if hpflag == 1 :
        SMask[xr] = 1
        Slabels[xr] = 0
        continue

    currentLength = len([SMask TMask])
    hminuserr = sum(abs(hneg.predict([XTrain(SMask == 1)XTrain(TMask == 1)]) - [Slabels(SMask == 1)Tlabels(TMask == 1)]))
    hpluserr = sum(abs(hplus.predict([XTrain(SMask == 1)XTrain(TMask == 1)]) - [Slabels(SMask == 1)Tlabels(TMask == 1)]))
    hminuserr /= (2 * currentLength)  # (1 - (-1)) = 2
    hpluserr /= (2 * currentLength)

    ###########################################
    # compute Delta DO NOT CHANGE THIS!
    delta = 0.01
    shatterCoeff = 2 * (t + 1)
    beta = sqrt((4 / t) * math.log(8 * (t ^ 2 + t) * shatterCoeff ^ 2 / delta))
    Delta = (beta ^ 2 + beta * (sqrt(hpluserr) + sqrt(hminuserr))) * .025
    ###########################################

    if hminuserr - hpluserr > Delta :
        SMask[xr] = 1
        Slabels[xr] = 1
        continue

    elif hpluserr - hminuserr > Delta :
        SMask[xr] = 1
        Slabels[xr] = 0
        continue

    TMask[xr] = 1
    Tlabels[xr] = YTrain[xr]
    cost += 1

    h, _ = subroutineSVM(XTrain(SMask[SMask == 1]), XTrain(TMask == 1), Slabels(SMask == 1), Tlabels(TMask == 1))
    SVMError = sum(abs(h.predict(XTest) - YTest)) / (2 * len(YTest))
    print('SVM error after #d rounds is #f'.format(), t, SVMError)

    xr = selectRandomUnlabeledPoint(R)
    R[xr] = 1
    hR, _ = subroutineSVM([], XTrain(R == 1), [], YTrain(R == 1))
    RandomError = sum(abs(hR.predict(XTest) - YTest)) / (2 * len(YTest))
    print('Random error after #d rounds is #f'.format(), t, RandomError)


