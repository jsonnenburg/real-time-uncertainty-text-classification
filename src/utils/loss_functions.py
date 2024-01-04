import tensorflow as tf


def null_loss(y_true, y_pred) -> tf.Tensor:
    return tf.zeros_like(y_true)


def aleatoric_loss(y_true, y_pred) -> tf.Tensor:
    """
    Aleatoric uncertainty loss function from Kendall & Gal (2017) for fine-tuning the teacher model.
    Does not require ground truth variance.
    """
    if not isinstance(y_pred, dict) or 'logits' not in y_pred or 'log_variances' not in y_pred:
        raise ValueError("y_pred must be a dictionary with 'logits' and 'log_variances' keys.")

    logits = y_pred['logits']
    log_variances = y_pred['log_variances']

    bce = tf.losses.BinaryCrossentropy(from_logits=True)
    cross_entropy_loss = bce(y_true, logits)

    precision = tf.exp(-log_variances)
    adjusted_loss = (precision * cross_entropy_loss) + log_variances

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
        raise KeyError("y_pred must be a dict with keys 'logits' and 'log_variances'.")


def shen_loss(y_true, y_pred) -> tf.Tensor:
    """
    Transfer learning loss function from Shen et al. (2021) for fine-tuning the student model.
    Weight corresponds to Lambda in the paper.

    y_true = Tuple(actual ground truth, teacher predictive sample)
    """
    weight = tf.convert_to_tensor(1, dtype=tf.float32)

    try:
        y_true, y_teacher = y_true[0], y_true[1]

        Lt = aleatoric_loss(y_true, y_pred)
        Ls = gaussian_mle_loss(y_teacher, y_pred)
        Ltotal = Ls + weight * Lt

        return tf.reduce_mean(Ltotal)
    except KeyError:
        raise KeyError("y_true must be a dict with keys 'labels' and 'predictions'.")


# shen loss function - implementation for classification in pytorch
# def shen_loss(y_true, y_pred):
#    Ls = K.mean(0.5*K.exp(-y_pred[:,1]) * K.pow(y_true[:,0] - y_pred[:,0],2) + 0.5*y_pred[:,1])
#    Lt = K.abs(y_true[:,0] - y_pred[:,0])
#    L = Ls + Lt
#    return K.mean(L)
