import numpy as np
from sklearn.svm import SVC
from util import select, stack


def subroutine_SVM(train_s, train_t, train_s_labels, train_t_labels):
    """
    This function learns a SVM model from training data. It takes two inputs, S and T. The algorithm
    tries to find a model consisent with S. If it can't, it returns a flag.
    If it can, it returns a model that minimizes the error on T, while still
    being consistent with S.
    :param train_s: training data with inferred labels, s x n
    :param train_t: training data with actual labels, t x n
    :param train_s_labels: labels for S, s x 1
    :param train_t_labels: labels for T, t x 1
    :return: [h, flag] learned SVM model, flag = 1 means inconsistent with s
    """

    flag = 1
    h = DefaultModel()

    if train_s.size == 0 and train_t.size == 0:
        assert train_s_labels.size == 0
        assert train_t_labels.size == 0
        # no data return default model, predicting all 0s
        flag = 0
        return h, flag
    elif train_s.size == 0:
        t, num_features = train_t.shape
        assert train_t_labels.shape == (t, 1)
        assert train_s_labels.size == 0

        flag = 0
        if select(train_t_labels, train_t_labels).size == 0:  # T has no positive samples
            return h, flag
        if select(train_t_labels, train_t_labels).size == train_t_labels.size: # T has no negative samples
            return PositiveModel(), flag
        h = SVC()
        h.fit(train_t, np.reshape(train_t_labels, train_t_labels.size))
        return h, flag
    elif train_t.size == 0:
        s, num_features = train_s.shape
        assert train_s_labels.shape == (s, 1)
        assert train_t_labels.size == 0

        if select(train_s_labels, train_s_labels).size == 0:
            flag = 0
        elif select(train_s_labels, train_s_labels).size == train_s_labels.size:  # all positive in S
            flag = 0
            h = PositiveModel()
        else:
            h = SVC()
            h.fit(train_s, np.reshape(train_s_labels, train_s_labels.size))
            predictions = h.predict(train_s)
            if len(predictions.shape) == 1:
                predictions = np.reshape(predictions, (predictions.size, 1))
            if np.sum(np.absolute(np.subtract(predictions, train_s_labels))) == 0:
                flag = 0
        return h, flag
    else:
        t, num_features = train_t.shape
        s, _ = train_s.shape
        assert train_s.shape == (s, num_features)
        assert train_t_labels.shape == (t, 1)
        assert train_s_labels.shape == (s, 1)

        if select(train_s_labels, train_s_labels).size == 0 and select(train_t_labels, train_t_labels).size == 0:
            flag = 0
        else:
            train_s_t = np.vstack((train_s, train_t))
            train_s_t_labels = np.vstack((train_s_labels, train_t_labels))
            assert train_s_t.shape == (s + t, num_features)
            assert train_s_t_labels.shape == (s + t, 1)

            h = SVC()
            h.fit(train_s_t, np.reshape(train_s_t_labels, train_s_t_labels.size))
            predictions = h.predict(train_s)
            if len(predictions.shape) == 1:
                predictions = np.reshape(predictions, (predictions.size, 1))
            if np.sum(np.absolute(np.subtract(predictions, train_s_labels))) == 0:
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
