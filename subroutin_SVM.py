import numpy as np
from sklearn.svm import SVC
from util import select, stack

def subroutine_SVM(S, T, S_labels, T_labels):
    """
    This function learns a SVM model from training data. It takes two inputs, S and T. The algorithm
    tries to find a model consisent with S. If it can't, it returns a flag.
    If it can, it returns a model that minimizes the error on T, while still
    being consistent with S.
    :param S: training data with inferred labels, s x n
    :param T: training data with actual labels, t x n
    :param S_labels: labels for S, s x 1
    :param T_labels: labels for T, t x 1
    :return: [h, flag] learned SVM model, flag = 1 means inconsistent with
    """
    flag = 1
    h = DefaultModel()

    if len(S) == 0 and len(T) == 0:
        flag = 0
        return h, flag
    elif len(S) == 0:
        flag = 0
        if not len(T_labels[T_labels == 1]) == 0:
            h = SVC()
            h.fit(T, T_labels)
        return h, flag
    elif len(T) == 0:
        if len(S_labels[S_labels == 1]) == 0:
            flag = 0
        elif len(S_labels[S_labels == 0]) == 0:
            flag = 0
            h = PositiveModel()
        else:
            h = SVC()
            h.fit(S, S_labels)
            if sum(abs(h.predict(S) - S_labels)) == 0:
                flag = 0
        return h, flag
    else:
        if len(S_labels[S_labels == 1]) == 0 and len(T_labels[T_labels == 1]) == 0:
            flag = 0
        else:
            h = SVC()
            h.fit(np.vstack((S, T)), np.vstack((S_labels, T_labels)))
            if sum(abs(h.predict(S) - S_labels)) == 0:
                flag = 0
        return h, flag


class DefaultModel:
    @classmethod
    def predict(cls, data):
        """
        default model returns -1 for all data
        :param data: np.array of m x n
        :return: np.appray of m x 1 with -1
        """
        return np.full((data.shape[0], 1), 0, dtype=np.int)


class PositiveModel:
    @classmethod
    def predict(cls, data):
        """
        default model returns -1 for all data
        :param data: np.array of m x n
        :return: np.appray of m x 1 with -1
        """
        return np.full((data.shape[0], 1), 1, dtype=np.int)
