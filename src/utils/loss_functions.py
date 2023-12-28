import tensorflow as tf

from logger_config import setup_logging
import logging

logger = logging.getLogger(__name__)

setup_logging()


def null_loss(y_true, y_pred) -> tf.Tensor:
    return tf.zeros_like(y_true)


def aleatoric_loss(y_true, y_pred) -> tf.Tensor:
    """
    Aleatoric uncertainty loss function from Kendall & Gal (2017) for fine-tuning the teacher model.
    Does not require ground truth variance.
    """
    # y_pred is assumed to contain both logits and log variance, concatenated
    # The first half of the dimensions are the logits, the second half the log variances
    bce = tf.losses.BinaryCrossentropy(from_logits=True)

    try:
        logits, log_variances = y_pred['logits'], y_pred['log_variances']
    except TypeError:
        return tf.convert_to_tensor(bce(y_true, y_pred.numpy().reshape(y_true.shape)).numpy())

    # TODO: if y_pred is ,1: return cross_entropy_loss

    # Standard cross-entropy loss between logits and true labels
    logits_np = logits.numpy().flatten()
    cross_entropy_loss = tf.convert_to_tensor(bce(y_true, logits_np))

    # Adjust cross-entropy loss by the predicted variance (aleatoric uncertainty)
    precision = tf.exp(-log_variances)
    adjusted_loss = precision * cross_entropy_loss + log_variances

    # The loss is the mean over all adjusted losses
    return tf.reduce_mean(adjusted_loss)


def gaussian_mle_loss(y_true, y_pred) -> tf.Tensor:
    """
    Gaussian MLE loss function from Shen et al. (2021) for fine-tuning the student model on the teacher's predictions.
    """
    try:
        logits, log_variances = y_pred['logits'], y_pred['log_variances']
        # calculate the Gaussian MLE loss
        # loss = 0.5 * tf.reduce_mean(variance_i + tf.square(y_true - mu_i) / variance_i - log_variance_i)
        loss = tf.reduce_mean(0.5 * tf.exp(-log_variances) * tf.square(y_true - logits) + 0.5 * log_variances)
        # TODO: check if this corresponds to Eq. 10 in Shen et al. (2021)
        return loss
    except KeyError:
        logger.error("y_pred must be a dict with keys 'logits' and 'log_variances'.")
        raise


def shen_loss(y_true, y_pred) -> tf.Tensor:
    """
    Transfer learning loss function from Shen et al. (2021) for fine-tuning the student model.
    Weight corresponds to Lambda in the paper.

    y_true = Tuple(actual ground truth, teacher predictive sample)
    """
    weight = tf.convert_to_tensor(1, dtype=tf.float32)

    try:
        y_true, y_teacher = y_true['labels'], y_true['predictions']

        Lt = aleatoric_loss(y_true, y_pred)
        Ls = gaussian_mle_loss(y_teacher, y_pred)
        Ltotal = Ls + weight * Lt

        return tf.reduce_mean(Ltotal)
    except KeyError:
        logger.error("y_true must be a dict with keys 'labels' and 'predictions'.")
        raise


# shen loss function - implementation for classification in pytorch
# def shen_loss(y_true, y_pred):
#    Ls = K.mean(0.5*K.exp(-y_pred[:,1]) * K.pow(y_true[:,0] - y_pred[:,0],2) + 0.5*y_pred[:,1])
#    Lt = K.abs(y_true[:,0] - y_pred[:,0])
#    L = Ls + Lt
#    return K.mean(L)
