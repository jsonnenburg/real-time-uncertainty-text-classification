import tensorflow as tf

# TODO: figure out output format of customized BERT model


# aleatoric uncertainty loss (Kendall & Gal, 2017)
def aleatoric_loss(y_true, y_pred):
    """Aleatoric uncertainty loss function from Kendall & Gal (2017) for fine-tuning the teacher model.
    Does not require ground truth variance.
    """
    # y_pred is assumed to contain both logits and log variance, concatenated
    # The first half of the dimensions are the logits, the second half the log variances
    logits, log_variances = tf.split(y_pred, num_or_size_splits=2, axis=1)

    # Standard cross-entropy loss between logits and true labels
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=logits)

    # Adjust cross-entropy loss by the predicted variance (aleatoric uncertainty)
    precision = tf.exp(-log_variances)
    adjusted_loss = precision * cross_entropy_loss + log_variances

    # The loss is the mean over all adjusted losses
    return tf.reduce_mean(adjusted_loss)


# student transfer learning loss (Shen et al., 2021)
def gaussian_mle_loss(y_true, y_pred_mean, y_pred_log_variance):
    """Gaussian MLE loss function from Shen et al. (2021) for fine-tuning the student model.
    """
    mu_i, log_variance_i = y_pred_mean, y_pred_log_variance

    # compute the exponential of the log variance to get the variance
    variance_i = tf.exp(log_variance_i)

    # calculate the Gaussian MLE loss
    loss = 0.5 * tf.reduce_mean(variance_i + tf.square(y_true - mu_i) / variance_i - log_variance_i)
    return loss


def shen_loss(y_true, y_pred, weight=1):
    """Transfer learning loss function from Shen et al. (2021) for fine-tuning the student model.
    Weight corresponds to Lambda in the paper.
    """
    assert y_pred.shape[1] == 2
    # unpack the predictions, this assumes that the predictions are a vector of size 2
    y_pred_mean, y_pred_log_variance = y_pred[:, 0], y_pred[:, 1]
    Lt = aleatoric_loss(y_true, y_pred_mean)
    Ls = gaussian_mle_loss(y_true, y_pred_mean, y_pred_log_variance)
    Ltotal = Ls + weight * Lt

    return Ltotal
