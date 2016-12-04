import numpy as np
from numpy import genfromtxt


def read_train_test(filename):
    """
    read train/test file format
    :param filename: train/test file
    :return: (feature_matrix, label_vector)
    """
    my_data = genfromtxt('data/{}'.format(filename), delimiter=',', dtype=np.dtype(np.int32))
    return my_data[:, :-1], my_data[:, -1]


def read_blind(filename):
    """
    read blind file format
    :param filename: blind file
    :return: (feature_matrix, id_vector)
    """
    my_data = genfromtxt('data/{}'.format(filename), delimiter=',', dtype=np.dtype(np.int32))
    return my_data[:, 1:], my_data[:, 1]


def write_prediction(filename, id_vector, prediction_vector):
    """
    write predictions to file
    :param filename: filename
    :param id_vector: id_vector
    :param prediction_vector: prediction vector
    :return: void
    """
    with open('predictions/{}'.format(filename), mode='w') as f:
        for line_id, prediction in zip(id_vector, prediction_vector):
            f.write('{line_id}, {prediction}\n'.format(**locals()))
