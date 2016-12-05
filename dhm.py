import math
import random

import numpy as np

from reader import read_train_test
from subroutin_SVM import subroutine_SVM
from util import select, stack

difficulty = 'EASY'


## select a random unlabeled point
def select_random_unlabeled_point(r):
    ur = np.where(r == 0)[0]  # get array of index of unlabeled points (random learning)
    xr = ur[random.randint(1, len(ur))-1]
    return xr

## This function runs the DHM and random learner in parallel assuming a streaming data model
# Input:  difficulty - the difficulty as a string, 'EAST' or "MODERATE'

# Additionally, you will implement a random learner for performing the
# same task and compare the performance of both algorithms

## ALGORITHM PARAMETERS
num_samples = 4000
# vectors for identifying points in sets S and T
S_mask = np.full((num_samples, ), 0, dtype=np.int)
T_mask = np.full((num_samples, ), 0, dtype=np.int)

# Labels for the points in S and T
S_labels = np.full((num_samples, ), 0, dtype=np.int)
T_labels = np.full((num_samples, ), 0, dtype=np.int)

# R_mask is a bit vector indicating which samples have been queried by a random learner
R_mask = np.full((num_samples, ), 0, dtype=np.int)
# generate the data.
#   XTrain is a 1 by num_samples vector of values in the interval [0,1].
#   YTrain is a 1 by num_samples vector of labels (either 0 or 1)
#   YTrain is the true model. You can use this to compute generalization error because abs(h-YTrain) = generalization error if h is the current model
X_train, y_train = read_train_test('{}_TRAIN.csv'.format(difficulty.upper()))
X_test, y_test = read_train_test('{}_TEST.csv'.format(difficulty.upper()))

# *************** IMPLEMENT THIS   ***************** #
# You may need to create local variables to keep track of sets S and T, etc
cost = 0
# this is the main loop of the DHM algorithm
for t in range(num_samples):

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
    h_neg, hn_flag = subroutine_SVM(stack(X_train, S_mask, X_train[t]), select(X_train, T_mask),
                                    stack(S_labels, S_mask, np.array([0])), select(T_labels, T_mask))
    if hn_flag == 1:
        S_mask[t] = 1
        S_labels[t] = 1
        continue

    h_plus, hp_flag = subroutine_SVM(stack(X_train, S_mask, X_train[t]), select(X_train, T_mask),
                                     stack(S_labels, S_mask, np.array([1])), select(T_labels, T_mask))
    if hp_flag == 1:
        S_mask[t] = 1
        S_labels[t] = 0
        continue

    currentLength = np.sum(S_mask) + np.sum(T_mask)
    hn_err = np.sum(abs(h_neg.predict(stack(X_train, S_mask, select(X_train, T_mask))) -
                    stack(S_labels, S_mask, select(T_labels, T_mask))))
    hp_err = np.sum(abs(h_plus.predict(stack(X_train, S_mask, select(X_train, T_mask))) -
                    stack(S_labels, S_mask, select(T_labels, T_mask))))
    hn_err /= currentLength + 1
    hp_err /= currentLength + 1

    ###########################################
    # compute Delta adapted from Homework
    new_idx = t + 1
    delta = 0.01
    shatter_coeff = 2 * (new_idx + 1)
    beta = np.sqrt((4 / new_idx) * math.log(8 * (np.power(new_idx, 2) + t) * np.power(shatter_coeff, 2) / delta))
    cap_delta = (np.power(beta, 2) + beta * (np.sqrt(hp_err) + np.sqrt(hn_err))) * .025
    ###########################################

    if hn_err - hp_err > cap_delta:
        S_mask[t] = 1
        S_labels[t] = 1
        continue

    elif hp_err - hn_err > cap_delta:
        S_mask[t] = 1
        S_labels[t] = 0
        continue

    T_mask[t] = 1
    T_labels[t] = y_train[t]
    cost += 1

    h, _ = subroutine_SVM(select(X_train, S_mask), select(X_train, T_mask), select(S_labels, S_mask), select(T_labels, T_mask))
    SVMError = np.sum(np.absolute(np.subtract(h.predict(X_test), y_test))) / y_test.size
    print('SVM error after {} rounds is {}'.format(t, SVMError))

    xr = select_random_unlabeled_point(R_mask)
    R_mask[xr] = 1
    hR, _ = subroutine_SVM(np.array([]), select(X_train, R_mask), np.array([]),
                           select(T_labels, R_mask))
    # hR, _ = subroutine_SVM(np.array([]), select(X_train, R_mask), np.array([]), select(y_train, R_mask))
    RandomError = np.sum(np.absolute(np.subtract(hR.predict(X_test), y_test))) / y_test.size
    print('Random error after {} rounds is {}'.format(t, RandomError))
