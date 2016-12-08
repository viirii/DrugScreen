import random
import sys


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from util_difficult import select
from sklearn.metrics import f1_score
from subroutin_SVM import DefaultModel

from reader import read_train_test, read_blind, write_prediction


# select a random unlabeled point
def select_random_unlabeled_point(mask):
    ur = np.where(mask == 0)[0]  # get array of index of unlabeled points (random learning)
    xr = ur[random.randint(1, len(ur))-1]
    return xr


def active_most_proba_random_forest(difficulty='DIFFICULT', num_init_label=500):
    num_init_label_copy = num_init_label
    current_model = None

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

    selected_label = np.full((num_samples, 1), -1, dtype=np.int)
    selected_mask = np.full((num_samples, 1), 0, dtype=np.int)

    # fill a base number of samples to selected
    for _ in range(num_init_label):
        x = select_random_unlabeled_point(selected_mask)
        selected_mask[x, 0] = 1
        selected_label[x, 0] = y_train[x, 0]

    # continue to fill until has at least a 1 and a 0 and a 2
    while not (np.any(selected_label == 0) and np.any(selected_label == 1) and np.any(selected_label == 2)):
        x = select_random_unlabeled_point(selected_mask)
        selected_mask[x, 0] = 1
        selected_label[x, 0] = y_train[x, 0]

    current_model = None
    r_label = np.full((num_samples, 1), -1, dtype=np.int)
    r_mask = np.full((num_samples, 1), 0, dtype=np.int)

    for _ in range(np.sum(selected_mask)):
        x = select_random_unlabeled_point(r_mask)
        r_mask[x, 0] = 1
        r_label[x, 0] = y_train[x, 0]

    blank_model = DefaultModel()
    b_predictions = blank_model.predict(X_test)

    # metrics needs to be recorded
    random_forest_errors = []
    random_errors = []
    blank_errors = []

    random_forest_f1s = []
    random_f1s = []
    blank_f1s = []
    t = np.sum(selected_mask)
    while np.sum(selected_mask) < 2500:
        t = np.sum(selected_mask)

        model = RandomForestClassifier(class_weight='balanced')
        labels_ = select(selected_label, selected_mask)
        model.fit(select(X_train, selected_mask), np.reshape(labels_, labels_.size))
        current_model = model

        predictions_with_proba = model.predict_proba(X_train)
        assert predictions_with_proba.shape == (num_samples, 3)

        classes = model.classes_
        assert classes.shape == (3, )
        pos_class_idx = np.where(classes == 1)[0][0]
        assert pos_class_idx == 0 or pos_class_idx == 1 or pos_class_idx == 2
        strong_class_idx = np.where(classes == 2)[0][0]
        assert strong_class_idx == 0 or strong_class_idx == 1 or strong_class_idx == 2

        max_proba = 0 # actually uses pos_proba * 1 + strong_proba * 2 for difficult
        max_idx = 0
        for i in range(num_samples):
            if selected_mask[i, 0] == 0:  # only consider unlabeled points
                proba = predictions_with_proba[i, pos_class_idx] + predictions_with_proba[i, strong_class_idx] * 2
                if proba > max_proba:
                    max_proba = proba
                    max_idx = i

        selected_mask[max_idx, 0] = 1
        selected_label[max_idx, 0] = y_train[max_idx, 0]

        predictions = model.predict(X_test)
        if len(predictions.shape) == 1:
            predictions = np.reshape(predictions, (predictions.size, 1))
        assert predictions.shape == (num_test, 1)
        random_forest_error = np.sum(np.absolute(np.subtract(predictions, y_test))) / y_test.size
        print('random_forest error after {} queries is {}'.format(t, random_forest_error))
        random_forest_errors.append(random_forest_error)
        random_forest_f1_score = f1_score(y_test > 0, predictions > 0)
        print('random_forest F1 after {} queries is {}'.format(t, random_forest_f1_score))
        random_forest_f1s.append(random_forest_f1_score)

        # Random selection Model
        xr = select_random_unlabeled_point(r_mask)
        r_mask[xr, 0] = 1
        r_label[xr, 0] = y_train[xr, 0]
        r = np.sum(r_mask)
        t = np.sum(selected_mask)
        if r != t:
            print("r = {}, t = {}".format(r, t))

        train_r = select(X_train, r_mask)
        train_r_label = select(y_train, r_mask)
        assert train_r.shape == (r, num_features)
        assert train_r_label.shape == (r, 1)

        model_r = RandomForestClassifier(class_weight='balanced')
        labels_ = select(r_label, r_mask)
        model_r.fit(select(X_train, r_mask), np.reshape(labels_, labels_.size))

        predictions = model_r.predict(X_test)
        if len(predictions.shape) == 1:
            predictions = np.reshape(predictions, (predictions.size, 1))
        assert predictions.shape == (num_test, 1)
        random_error = np.sum(np.absolute(np.subtract(predictions, y_test))) / y_test.size
        print('Random error after {} queries is {}'.format(r, random_error))
        random_errors.append(random_error)
        random_f1_score = f1_score(y_test > 0, predictions > 0)
        print('Random F1 after {} queries is {}'.format(t, random_f1_score))
        random_f1s.append(random_f1_score)

        # Blank Model (prediction all negative from the start)
        blank_error = np.sum(np.absolute(np.subtract(b_predictions, y_test))) / y_test.size
        print('Blank learner error queries is {}'.format(blank_error))
        blank_errors.append(blank_error)
        blank_f1_score = f1_score(y_test > 0, b_predictions > 0)
        print('Blank F1 after {} queries is {}'.format(t, blank_f1_score))
        blank_f1s.append(random_f1_score)

    # Final writings
    predictions = current_model.predict(X_test)
    if len(predictions.shape) == 1:
        predictions = np.reshape(predictions, (predictions.size, 1))
    final_error = np.sum(np.absolute(np.subtract(predictions, y_test))) / y_test.size
    print('final random_forest error is {}'.format(final_error))
    final_f1_score = f1_score(y_test, predictions)
    print('final random_forest F1 is {}'.format(final_f1_score))
    print('final number of queries is'.format(t))

    feature_matrix, id_vector = read_blind('{}_BLINDED.csv'.format(difficulty.upper()))
    blinded_predictions = current_model.predict(feature_matrix)
    blinded_predictions = np.reshape(blinded_predictions, blinded_predictions.size)
    write_prediction('AMRF_{}_BLINDED_PREDICTION_{}.csv'.format(difficulty.upper(), num_init_label_copy),
                     id_vector, blinded_predictions)

    with open('output/AMRF_{}_metrics_{}.txt'.format(difficulty.upper(), num_init_label_copy), mode='w') as f:
        f.write('random_forest errors\n')
        f.write(random_forest_errors.__str__())
        f.write('\n')
        f.write('Random errors\n')
        f.write(random_errors.__str__())
        f.write('\n')
        f.write('Blank errors\n')
        f.write(blank_errors.__str__())
        f.write('\n')

        f.write('random_forest F1 scores\n')
        f.write(random_forest_f1s.__str__())
        f.write('\n')
        f.write('Random F1 scores\n')
        f.write(random_f1s.__str__())
        f.write('\n')
        f.write('Blank F1 scores\n')
        f.write(blank_f1s.__str__())
        f.write('\n')

        f.flush()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        active_most_proba_random_forest(difficulty='DIFFICULT', num_init_label=500)
    if len(sys.argv) == 2:
        active_most_proba_random_forest(difficulty='DIFFICULT', num_init_label=int(sys.argv[1]))
    else:
        active_most_proba_random_forest(difficulty=sys.argv[1], num_init_label=int(sys.argv[2]))
