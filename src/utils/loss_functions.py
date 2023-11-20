import tensorflow as tf

# for binary classification with sigmoid activation, this is equivalent to categorical cross-entropy
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# student transfer learning loss (Shen et al., 2021)
def gaussian_mle_loss(y_true, y_pred_mean, y_pred_variance):
    mu_i, log_variance_i = y_pred_mean, y_pred_variance

    # compute the exponential of the log variance to get the variance
    variance_i = tf.exp(log_variance_i)

    # calculate the Gaussian MLE loss
    loss = 0.5 * tf.reduce_mean(variance_i + tf.square(y_true - mu_i) / variance_i - log_variance_i)
    return loss


def shen_loss(y_true, y_pred, weight=1):
    assert y_pred.shape[1] == 2
    # unpack the predictions, this assumes that the predictions are a vector of size 2
    y_pred_mean, y_pred_variance = y_pred[:, 0], y_pred[:, 1]
    Lt = bce(y_true, y_pred_mean)
    Ls = gaussian_mle_loss(y_true, y_pred_mean, y_pred_variance)
    Ltotal = Ls + weight * Lt

    return Ltotal
