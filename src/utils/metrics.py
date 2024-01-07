import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from scipy.stats import entropy

# y_pred: need class predictions
# y_prob: need probabilities for positive class


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0 if numerator == 0 else np.nan
    else:
        return numerator / denominator


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return safe_divide(true_positives, predicted_positives)


def recall_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return safe_divide(true_positives, actual_positives)


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return safe_divide(2 * precision * recall, precision + recall)


def nll_score(y_true, y_prob):
    y_true = np.squeeze(y_true)
    return log_loss(y_true, y_prob)


def brier_score(y_true, y_prob):
    y_true = np.squeeze(y_true)
    return brier_score_loss(y_true, y_prob)


def pred_entropy_score(y_probs):
    p = np.stack([1 - y_probs, y_probs], axis=-1)
    log_p = np.log2(np.clip(p, 1e-9, 1))
    entropy = -np.sum(p * log_p, axis=-1)
    return entropy


def ece_score(y_true, y_pred, y_prob, n_bins=10):
    y_true = np.squeeze(y_true)
    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.greater_equal(y_prob, bin_lower) & np.less(y_prob, bin_upper)
        in_bin = in_bin.flatten()
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_pred[in_bin] == y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece
