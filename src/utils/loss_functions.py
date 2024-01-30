import tensorflow as tf


def null_loss(y_true, y_pred) -> tf.Tensor:
    """
    Null loss function (dummy loss) for fine-tuning the custom aleatoric uncertainty BERT models.
    """
    return tf.zeros_like(y_true)


# The following was adapted from https://github.com/kyle-dorman/bayesian-neural-network-blogpost/tree/master.
def bayesian_binary_crossentropy(T):
    """
    The aleatoric loss function as defined by Kendall & Gal (2017) in Eq. 10 for fine-tuning the aleatoric uncertainty BERT teacher.

    :param T: The number of Monte Carlo samples to take.
    """
    def bayesian_binary_crossentropy_internal(y_true, y_pred):
        y_true = tf.expand_dims(y_true, -1)
        pred = y_pred['logits']
        log_variance = y_pred['log_variances']
        variance = tf.exp(log_variance)
        std = tf.sqrt(variance)

        variance_depressor = tf.exp(variance) - tf.ones_like(variance)
        undistorted_loss = tf.keras.losses.binary_crossentropy(y_true, pred, from_logits=True)

        def monte_carlo_fn(i):
            return gaussian_binary_crossentropy(y_true, pred, std, undistorted_loss)

        monte_carlo_results = tf.map_fn(
            monte_carlo_fn,
            tf.ones(T),
            fn_output_signature=tf.float32
        )
        variance_loss = tf.reduce_mean(monte_carlo_results, axis=0) * undistorted_loss

        return variance_loss + undistorted_loss + variance_depressor

    return bayesian_binary_crossentropy_internal


# Gaussian binary cross entropy
def gaussian_binary_crossentropy(true, pred, std, undistorted_loss):
    std_sample = tf.random.normal(shape=tf.shape(true), stddev=std)
    distorted_loss = tf.keras.losses.binary_crossentropy(true, pred + std_sample, from_logits=True)
    diff = undistorted_loss - distorted_loss
    return -tf.nn.elu(diff)


def bce_loss(y_true, y_pred) -> tf.Tensor:
    """
    Binary cross entropy loss function for fine-tuning the student model.
    """
    y_true = tf.expand_dims(y_true, -1)
    try:
        logits = y_pred['mean_logits']
        loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(y_true, logits, from_logits=True))
        return loss
    except KeyError:
        raise KeyError("y_pred must be a dict with key 'mean_logits'.")


def gaussian_mle_loss(y_true, y_pred, n_samples: int) -> tf.Tensor:
    """
    Gaussian MLE loss function from Shen et al. (2021) for fine-tuning the student model on the teacher's predictions.

    :param n_samples: The number of predictive samples generated for the teacher model, as determined by m and k.
    """
    logits, log_variances = y_pred['logits'], y_pred['log_variances']

    logits = tf.tile(logits, tf.constant([1, n_samples], dtype=tf.int32))
    log_variances = tf.tile(log_variances, tf.constant([1, n_samples], dtype=tf.int32))

    loss = tf.reduce_mean(0.5 * tf.exp(-log_variances) * tf.norm(y_true - logits, ord='euclidean') + 0.5 * log_variances)

    return loss


def shen_loss(loss_weight: int = 1, n_samples: int = 50):
    def shen_loss_internal(y_true, y_pred) -> tf.Tensor:
        """
        Transfer learning loss function from Shen et al. (2021) for fine-tuning the student model.
        Weight corresponds to Lambda in the paper.

        y_true = Tuple(actual ground truth - CLASS LABELS, teacher predictive sample - LOGITS)
        y_pred = dict with keys 'logits' and 'log_variances', the outputs of the student model heads
        """
        weight = tf.convert_to_tensor(loss_weight, dtype=tf.float32)

        y_true, y_teacher = y_true[0], y_true[1]

        Lt = bce_loss(y_true, y_pred)
        Ls = gaussian_mle_loss(y_teacher, y_pred, n_samples)
        Ltotal = Ls + weight * Lt

        return Ltotal

    return shen_loss_internal
