import tensorflow as tf


def null_loss(y_true, y_pred):
    return tf.zeros_like(y_true)


def aleatoric_loss(y_true, y_pred):
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


def gaussian_mle_loss(y_true, y_pred_mean, y_pred_log_variance):
    """
    Gaussian MLE loss function from Shen et al. (2021) for fine-tuning the student model on the teacher's predictions.
    """
    mu_i, log_variance_i = y_pred_mean, y_pred_log_variance

    # calculate the Gaussian MLE loss
    # loss = 0.5 * tf.reduce_mean(variance_i + tf.square(y_true - mu_i) / variance_i - log_variance_i)
    loss = tf.reduce_mean(0.5 * tf.exp(-log_variance_i) * tf.square(y_true - mu_i) + 0.5 * log_variance_i)
    # TODO: check if this corresponds to Eq. 10 in Shen et al. (2021)
    return loss


def shen_loss(y_true, y_pred, weight=1):
    """
    Transfer learning loss function from Shen et al. (2021) for fine-tuning the student model.
    Weight corresponds to Lambda in the paper.

    y_true = Tuple(actual ground truth, teacher predictive sample)
    """
    # TODO: update to use y_pred dict and and proper tf-compliant losses (see aleatoric loss)

    assert y_true.shape[1] == 2 and y_pred.shape[1] == 2

    y_true, y_teacher_pred_mean = y_true[:, 0], y_true[:, 1]
    # unpack the predictions, this assumes that the predictions are a vector of size 2
    y_student_pred_mean, y_student_pred_log_variance = y_pred[:, 0], y_pred[:, 1]
    Lt = aleatoric_loss(y_true, y_pred)
    Ls = gaussian_mle_loss(y_teacher_pred_mean, y_student_pred_mean, y_student_pred_log_variance)
    Ltotal = Ls + weight * Lt

    return tf.reduce_mean(Ltotal)

# shen loss function - implementation for classification in pytorch
# def shen_loss(y_true, y_pred):
#    Ls = K.mean(0.5*K.exp(-y_pred[:,1]) * K.pow(y_true[:,0] - y_pred[:,0],2) + 0.5*y_pred[:,1])
#    Lt = K.abs(y_true[:,0] - y_pred[:,0])
#    L = Ls + Lt
#    return K.mean(L)
