import math
import random
import sys

import numpy as np
from sklearn.metrics import f1_score

from reader import read_train_test, read_blind, write_prediction
from subroutin_SVM import subroutine_SVM, DefaultModel
from util import select, stack


# select a random unlabeled point
def select_random_unlabeled_point(mask):
    ur = np.where(mask == 0)[0]  # get array of index of unlabeled points (random learning)
    xr = ur[random.randint(1, len(ur))-1]
    return xr


def dhm(difficulty='EASY', num_init_label=500):
    num_init_label_copy = num_init_label
    current_model = None
    t = 0
    # This function runs the DHM and random learner in parallel assuming a streaming data model
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


    # fill a base number of samples to T
    for _ in range(num_init_label):
        x = select_random_unlabeled_point(T_mask)
        T_mask[x, 0] = 1
        T_labels[x, 0] = y_train[x, 0]

    # R_mask is a bit vector indicating which samples have been queried by a random learner
    R_mask = np.full((num_samples, 1), 0, dtype=np.int)

    for _ in range(num_init_label):
        x = select_random_unlabeled_point(R_mask)
        R_mask[x, 0] = 1

    n_quires = np.full((num_samples, 1), 0, dtype=np.int)

    # Blank learner is the one which predicts negative label all the time
    hB = DefaultModel()
    B_predictions = hB.predict(X_test)

    # number of queries per round
    queries = [0 for _ in range(num_samples)]

    # metrics needs to be recorded
    svm_errors = []
    random_errors = []
    blank_errors = []

    svm_f1s = []
    random_f1s = []
    blank_f1s = []

    # this is the main loop of the DHM algorithm
    for x in range(num_samples):
        if T_mask[x, 0] == 1:  # label already fed to T
            num_init_label -= 1  # the label now is considered fed by the for loop
            continue

        print("round {}".format(x))

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
        n_quires[x, 0] = t
        assert x == s + t - num_init_label

        train_s = stack(X_train, S_mask, next_instance)
        train_t = select(X_train, T_mask)
        train_s_label = stack(S_labels, S_mask, np.full((1, 1), 0, dtype=np.int))
        train_t_label = select(T_labels, T_mask)
        assert train_s.shape == (s + 1, num_features)
        assert train_t.shape == (t, num_features)
        assert train_s_label.shape == (s + 1, 1)
        assert train_t_label.shape == (t, 1)

        if current_model is not None:
            predictions = current_model.predict(train_s)
            if len(predictions.shape) == 1:
                predictions = np.reshape(predictions, (predictions.size, 1))
            s_error = np.sum(np.absolute(np.subtract(predictions, train_s_label))) / y_test.size
            print('current s error is {}'.format(s_error))

        h_neg, hn_flag = subroutine_SVM(train_s, train_t, train_s_label, train_t_label)

        train_s_label = stack(S_labels, S_mask, np.full((1, 1), 1, dtype=np.int))
        assert train_s_label.shape == (s + 1, 1)

        h_pos, hp_flag = subroutine_SVM(train_s, train_t, train_s_label, train_t_label)

        if hn_flag == 1:
            print("Only positive works")
            S_mask[x, 0] = 1
            S_labels[x, 0] = 1
            current_model = h_pos
            continue

        if hp_flag == 1:
            print("Only negative works")
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
            print("Positive has lower error")
            S_mask[x, 0] = 1
            S_labels[x, 0] = 1
            current_model = h_pos
            continue

        elif hp_err - hn_err > cap_delta:
            print("Negative has lower error")
            S_mask[x, 0] = 1
            S_labels[x, 0] = 0
            current_model = h_neg
            continue

        # other wise add current line to T
        T_mask[x, 0] = 1
        T_labels[x, 0] = y_train[x, 0]

        s = np.sum(S_mask)
        t = np.sum(T_mask)
        assert x + 1 == s + t - num_init_label

        train_s = select(X_train, S_mask)
        train_t = select(X_train, T_mask)
        train_s_label = select(S_labels, S_mask)
        train_t_label = select(T_labels, T_mask)
        assert (s == 0 and train_s.size == 0) or train_s.shape == (s, num_features)
        assert train_t.shape == (t, num_features)
        assert (s == 0 and train_s_label.size == 0) or train_s_label.shape == (s, 1)
        assert train_t_label.shape == (t, 1)

        h, _ = subroutine_SVM(train_s, train_t, train_s_label, train_t_label)
        predictions = h.predict(X_test)
        if len(predictions.shape) == 1:
            predictions = np.reshape(predictions, (predictions.size, 1))
        assert predictions.shape == (num_test, 1)
        SVMError = np.sum(np.absolute(np.subtract(predictions, y_test))) / y_test.size
        print('SVM error after {} queries is {}'.format(t, SVMError))
        svm_errors.append(SVMError)
        queries[x] = t
        svm_f1_score = f1_score(y_test, predictions)
        print('SVM F1 after {} queries is {}'.format(t, svm_f1_score))
        svm_f1s.append(svm_f1_score)

        # Random selection Model
        xr = select_random_unlabeled_point(R_mask)
        R_mask[xr, 0] = 1
        r = np.sum(R_mask)
        assert r == t

        train_r = select(X_train, R_mask)
        train_r_label = select(y_train, R_mask)
        assert train_r.shape == (r, num_features)
        assert train_r_label.shape == (r, 1)

        hR, _ = subroutine_SVM(np.zeros((0, num_features)), train_r, np.zeros((0, 1)), train_r_label)
        predictions = hR.predict(X_test)
        if len(predictions.shape) == 1:
            predictions = np.reshape(predictions, (predictions.size, 1))
        assert predictions.shape == (num_test, 1)
        random_error = np.sum(np.absolute(np.subtract(predictions, y_test))) / y_test.size
        print('Random error after {} queries is {}'.format(r, random_error))
        random_errors.append(random_error)
        random_f1_score = f1_score(y_test, predictions)
        print('Random F1 after {} queries is {}'.format(t, random_f1_score))
        random_f1s.append(random_f1_score)

        # Blank Model (prediction all negative from the start)
        blank_error = np.sum(np.absolute(np.subtract(B_predictions, y_test))) / y_test.size
        print('Blank learner error queries is {}'.format(blank_error))
        blank_errors.append(blank_error)
        blank_f1_score = f1_score(y_test, B_predictions)
        print('Blank F1 after {} queries is {}'.format(t, blank_f1_score))
        blank_f1s.append(random_f1_score)

        if t > 2500:
            break

    predictions = current_model.predict(X_test)
    if len(predictions.shape) == 1:
        predictions = np.reshape(predictions, (predictions.size, 1))
    final_error = np.sum(np.absolute(np.subtract(predictions, y_test))) / y_test.size
    print('final SVM error is {}'.format(final_error))
    final_f1_score = f1_score(y_test, predictions)
    print('final SVM F1 is {}'.format(final_f1_score))
    print('final number of queries is'.format(t))

    feature_matrix, id_vector = read_blind('{}_BLINDED.csv'.format(difficulty.upper()))
    blinded_predictions = current_model.predict(feature_matrix)
    blinded_predictions = np.reshape(blinded_predictions, blinded_predictions.size)
    write_prediction('{}_BLINDED_PREDICTION_{}.csv'.format(difficulty.upper(), num_init_label_copy),
                     id_vector, blinded_predictions)

    with open('output/{}_metrics_{}.txt'.format(difficulty.upper(), num_init_label_copy), mode='w') as f:
        f.write('SVM errors\n')
        f.write(svm_errors.__str__())
        f.write('\n')
        f.write('Random errors\n')
        f.write(random_errors.__str__())
        f.write('\n')
        f.write('Blank errors\n')
        f.write(blank_errors.__str__())
        f.write('\n')

        f.write('SVM F1 scores\n')
        f.write(svm_f1s.__str__())
        f.write('\n')
        f.write('Random F1 scores\n')
        f.write(random_f1s.__str__())
        f.write('\n')
        f.write('Blank F1 scores\n')
        f.write(blank_f1s.__str__())
        f.write('\n')

        f.write('# queries per round\n')
        f.write(np.reshape(n_quires, n_quires.size).__str__())
        f.write('\n')

        f.flush()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        dhm(difficulty='EASY', num_init_label=500)
    else:
        dhm(difficulty=sys.argv[1], num_init_label=int(sys.argv[2]))
