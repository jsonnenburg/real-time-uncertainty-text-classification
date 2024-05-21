import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple


def json_serialize(value):
    if np.isscalar(value):
        if np.isnan(value):
            return 'NaN'
        elif isinstance(value, np.ndarray) or isinstance(value, tf.Tensor):
            return value.item()
        elif type(value) is np.float32:
            return value.item()
        else:
            return value
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return value


def safe_divide(numerator, denominator) -> float:
    if denominator == 0:
        return 0 if numerator == 0 else np.nan
    else:
        return numerator / denominator


def accuracy_score(y_true, y_pred) -> float:
    """
    :param y_true: The true class labels.
    :param y_pred: The predicted class labels.
    """
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred) -> float:
    """
    :param y_true: The true class labels.
    :param y_pred: The predicted class labels.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return safe_divide(true_positives, predicted_positives)


def recall_score(y_true, y_pred) -> float:
    """
    :param y_true: The true class labels.
    :param y_pred: The predicted class labels.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return safe_divide(true_positives, actual_positives)


def f1_score(y_true, y_pred) -> float:
    """
    :param y_true: The true class labels.
    :param y_pred: The predicted class labels.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return safe_divide(2 * precision * recall, precision + recall)


def auc_score(y_true, y_prob) -> float:
    """
    :param y_true: The true class labels.
    :param y_prob: The predicted probabilities of the positive class.
    """
    return roc_auc_score(y_true, y_prob)


def nll_score(y_true, y_prob) -> float:
    """
    :param y_true: The true class labels.
    :param y_prob: The predicted probabilities of the positive class.
    """
    y_true = np.squeeze(y_true)
    return log_loss(y_true, y_prob)


def brier_score(y_true, y_prob) -> float:
    """
    :param y_true: The true class labels.
    :param y_prob: The predicted probabilities of the positive class.
    """
    y_true = np.squeeze(y_true)
    return brier_score_loss(y_true, y_prob)


def adapt_logits_for_binary_classification(logits) -> tf.Tensor:
    """
    Adapts logits from shape (N, 1) to (N, 2) for binary classification.

    :param logits: The logits for the positive class.
    """
    # Logits for the negative class could be inferred as zeros or the negation of the positive logits
    # Here, we use negation to construct complementary logits for the two classes
    negative_class_logits = -logits  # Assuming logits represent the positive class
    adapted_logits = tf.concat([negative_class_logits, logits], axis=1)
    return adapted_logits


def brier_score_decomposition(y_true, y_pred_logits) -> Tuple[float, float, float]:
    """
    Computes the Brier score decomposition into uncertainty, resolution, and reliability components.
    :param y_true: The true class labels.
    :param y_pred_logits: The predicted logits for the positive class.
    """
    y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.int32)
    logits_tensor = tf.convert_to_tensor(y_pred_logits, dtype=tf.float32)

    # turn y_pred_logits from (N,) to (N, 1)
    logits_tensor = tf.reshape(logits_tensor, [-1, 1])

    # Adapt logits from shape (N, 1) to (N, 2)
    adapted_logits_tensor = adapt_logits_for_binary_classification(logits_tensor)

    # Ensure y_true is a tensor of shape (n_samples,)
    y_true_tensor = tf.squeeze(y_true_tensor)

    uncertainty, resolution, reliability = tfp.stats.brier_decomposition(y_true_tensor, adapted_logits_tensor)
    return uncertainty.numpy(), resolution.numpy(), reliability.numpy()


def pred_entropy_score(y_probs) -> np.ndarray:
    """
    Computes the entropy of the predicted probabilities.
    :param y_probs: The predicted probabilities of the positive class.
    """
    p = np.stack([1 - y_probs, y_probs], axis=-1)
    log_p = np.log2(np.clip(p, 1e-9, 1))
    entropy = -np.sum(p * log_p, axis=-1)
    return entropy


def ece_score(y_true, y_pred, y_prob, n_bins=30):
    """
    Computes the expected calibration error with a default bin size of 30 and using the L2 norm,
    as used by Shen et al. (2021).
    """
    y_true = np.squeeze(y_true)

    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]

    sum_sq_diff = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.greater_equal(y_prob, bin_lower) & np.less(y_prob, bin_upper)
        in_bin = in_bin.flatten()
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_pred[in_bin] == y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            sum_sq_diff += ((avg_confidence_in_bin - accuracy_in_bin) ** 2) * prop_in_bin

    ece_l2 = np.sqrt(sum_sq_diff)

    return ece_l2


def ece_score_l1_tfp(y_true, y_pred_logits, n_bins=10) -> float:
    """
    Computes the Expected Calibration Error (ECE) using the L1 norm.

    :param y_true: The true class labels.
    :param y_pred_logits: The predicted logits for the positive class.
    :param n_bins: The number of bins to use for the ECE computation.
    """
    y_true_tensor = tf.convert_to_tensor(y_true, dtype=tf.int32)
    logits_tensor = tf.convert_to_tensor(y_pred_logits, dtype=tf.float32)

    # turn y_pred_logits from (N,) to (N, 1)
    logits_tensor = tf.reshape(logits_tensor, [-1, 1])

    # Adapt logits from shape (N, 1) to (N, 2)
    adapted_logits_tensor = adapt_logits_for_binary_classification(logits_tensor)

    # Ensure y_true is a tensor of shape (n_samples,)
    y_true_tensor = tf.squeeze(y_true_tensor)

    ece = tfp.stats.expected_calibration_error(n_bins, logits=adapted_logits_tensor, labels_true=y_true_tensor)

    return ece.numpy()


def bald_score(y_prob_mc) -> np.ndarray:
    """
    Computes the BALD score as defined by Houlsby et al. (2011) for evaluating the predictive uncertainty of a model.
    Defined as the difference between the entropy of the predictive distribution and the expected entropy of the
    predictive distribution. The expected entropy is approximated by the entropy of the mean predictive distribution.

    :param y_prob_mc: The probability predictions over Monte Carlo or Dropout samples.

    Predictive Entropy: This is the entropy of the predictive distribution, which quantifies the uncertainty in the
    model's predictions. It's typically calculated as the entropy of the averaged predictions over multiple stochastic
    forward passes (with dropout enabled in the case of BERT or similar models).

    Expected Data Entropy: This is the average of the entropy of the predictions for each stochastic forward pass.
    It measures the average uncertainty in the predictions for each individual pass.
    """
    def compute_entropy(probs):
        return -np.sum(probs * np.log(probs + 1e-10), axis=-1)

    y_prob_mc_mean = np.mean(y_prob_mc, axis=1)

    # Predictive Entropy - Entropy of the mean predictive distribution
    predictive_entropy = compute_entropy(y_prob_mc_mean)

    # Expected Data Entropy - Mean of entropies of predictions for each stochastic forward pass
    entropies_per_pass = compute_entropy(y_prob_mc)
    expected_data_entropy = np.mean(entropies_per_pass, axis=1)

    # BALD score - Difference between Predictive Entropy and Expected Data Entropy
    bald = predictive_entropy - expected_data_entropy

    return bald
