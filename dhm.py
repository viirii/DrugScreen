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

current_model = None
## This function runs the DHM and random learner in parallel assuming a streaming data model
# Input:  difficulty - the difficulty as a string, 'EAST' or "MODERATE'

# Additionally, you will implement a random learner for performing the
# same task and compare the performance of both algorithms

# generate the data.
#   XTrain is a 1 by num_samples vector of values in the interval [0,1].
#   YTrain is a 1 by num_samples vector of labels (either 0 or 1)
#   YTrain is the true model
X_train, y_train = read_train_test('{}_TRAIN.csv'.format(difficulty.upper()))
X_test, y_test = read_train_test('{}_TEST.csv'.format(difficulty.upper()))

num_samples = X_train.shape[0]
num_test = X_test.shape[0]
num_features = X_train.shape[1]
assert y_train.shape == (num_samples, 1)
assert X_test.shape == (num_test, num_features)
assert y_test.shape == (num_test, 1)

# vectors for identifying points in sets S and T
S_mask = np.full((num_samples, 1), 0, dtype=np.int)
T_mask = np.full((num_samples, 1), 0, dtype=np.int)

# Labels for the points in S and T
S_labels = np.full((num_samples, 1), 0, dtype=np.int)
T_labels = np.full((num_samples, 1), 0, dtype=np.int)

# R_mask is a bit vector indicating which samples have been queried by a random learner
R_mask = np.full((num_samples, ), 0, dtype=np.int)


# *************** IMPLEMENT THIS   ***************** #
# You may need to create local variables to keep track of sets S and T, etc
cost = 0
# this is the main loop of the DHM algorithm
for x in range(num_samples):

    # XTrain(t) is the next instance in the data stream
    next_instance = X_train[[x], :]
    assert next_instance.shape == (1, num_features)
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
    s = np.sum(S_mask)
    t = np.sum(T_mask)
    assert x == s + t

    train_s = stack(X_train, S_mask, next_instance)
    train_t = select(X_train, T_mask)
    train_s_label = stack(S_labels, S_mask, np.full((1, 1), 0, dtype=np.int))
    train_t_label = select(T_labels, T_mask)
    assert train_s.shape == (s + 1, num_features)
    assert train_t.shape == (t, num_features)
    assert train_s_label.shape == (s + 1, 1)
    assert train_t_label.shape == (t, 1)

    h_neg, hn_flag = subroutine_SVM(train_s, train_t, train_s_label, train_t_label)

    train_s_label = stack(S_labels, S_mask, np.full((1, 1), 1, dtype=np.int))
    assert train_s_label.shape == (s + 1, 1)

    h_pos, hp_flag = subroutine_SVM(train_s, train_t, train_s_label, train_t_label)

    if hn_flag == 1:
        S_mask[x, 0] = 1
        S_labels[x, 0] = 1
        current_model = h_pos
        continue

    if hp_flag == 1:
        S_mask[x, 0] = 1
        S_labels[x, 0] = 0
        current_model = h_neg
        continue

    train_s_t = stack(X_train, S_mask, select(X_train, T_mask))
    train_s_t_label = stack(S_labels, S_mask, select(T_labels, T_mask))
    assert train_s_t.shape == (s + t, num_features)
    assert train_s_t_label.shape == (s + t, 1)

    hn_err = np.sum(np.absolute(np.subtract(h_neg.predict(train_s_t), train_s_t_label)))
    hp_err = np.sum(np.absolute(np.subtract(h_pos.predict(train_s_t), train_s_t_label)))
    hn_err /= x + 1
    hp_err /= x + 1

    ###########################################
    # compute Delta adapted from Homework
    new_idx = x + 1  # to avoid division by 0
    delta = 0.01
    shatter_coeff = 2 * (new_idx + 1)
    beta = np.sqrt((4 / new_idx) * math.log(8 * (np.power(new_idx, 2) + x) * np.power(shatter_coeff, 2) / delta))
    cap_delta = (np.power(beta, 2) + beta * (np.sqrt(hp_err) + np.sqrt(hn_err))) * .025
    ###########################################

    if hn_err - hp_err > cap_delta:
        S_mask[x, 0] = 1
        S_labels[x, 0] = 1
        current_model = h_pos
        continue

    elif hp_err - hn_err > cap_delta:
        S_mask[x, 0] = 1
        S_labels[x, 0] = 0
        current_model = h_neg
        continue

    # other wise add current line to T
    T_mask[x, 0] = 1
    T_labels[x, 0] = y_train[x, 0]

    s = np.sum(S_mask)
    t = np.sum(T_mask)
    assert x + 1 == s + t

    train_s = select(X_train, S_mask)
    train_t = select(X_train, T_mask)
    train_s_label = select(S_labels, S_mask)
    train_t_label = select(T_labels, T_mask)
    assert (s == 0 and train_s.size == 0) or train_s.shape == (s, num_features)
    assert train_t.shape == (t, num_features)
    assert (s == 0 and train_s_label.size == 0) or train_s_label.shape == (s + 1, 1)
    assert train_t_label.shape == (t, 1)

    h, _ = subroutine_SVM(train_s, train_t, train_s_label, train_t_label)
    SVMError = np.sum(np.absolute(np.subtract(h.predict(X_test), y_test))) / y_test.size
    print('SVM error after {} rounds is {}'.format(x, SVMError))

    xr = select_random_unlabeled_point(R_mask)
    R_mask[xr] = 1
    r = np.sum(R_mask)
    assert r == t

    train_r = select(X_train, R_mask)
    train_r_label = select(y_train, R_mask)
    assert train_r.shape == (r, num_features)
    assert train_r_label.shape == (r, 1)

    hR, _ = subroutine_SVM(np.zeros((0, num_features)), train_r, np.zeros((0, 1)), train_r_label)
    # hR, _ = subroutine_SVM(np.array([]), select(X_train, R_mask), np.array([]), select(y_train, R_mask))
    RandomError = np.sum(np.absolute(np.subtract(hR.predict(X_test), y_test))) / y_test.size
    print('Random error after {} rounds is {}'.format(x, RandomError))
