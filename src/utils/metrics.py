import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from scipy.stats import entropy


# TODO: change y_pred to y_prob

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    return np.mean(y_true[y_pred == 1] == y_pred[y_pred == 1])


def recall_score(y_true, y_pred):
    return np.mean(y_true[y_true == 1] == y_pred[y_true == 1])


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


def nll_score(y_true, y_prob):
    return log_loss(y_true, y_prob)


def brier_score(y_true, y_prob):
    return brier_score_loss(y_true, y_prob[:, 1])


def pred_entropy_score(y_prob):
    return entropy(y_prob.T)


def ece_score(y_true, y_prob, n_bins=10):
    # TODO: write unit test
    y_pred = np.argmax(y_prob, axis=1)
    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.greater_equal(y_prob[:, 1], bin_lower) & np.less(y_prob[:, 1], bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_pred[in_bin] == y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin, 1])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
