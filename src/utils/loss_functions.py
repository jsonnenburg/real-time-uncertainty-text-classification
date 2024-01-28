import tensorflow as tf


def null_loss(y_true, y_pred) -> tf.Tensor:
    """
    Null loss function (dummy loss) for fine-tuning the custom aleatoric uncertainty BERT models.
    """
    return tf.zeros_like(y_true)


def archived_aleatoric_loss(y_true, y_pred) -> tf.Tensor:
    """
    Aleatoric uncertainty loss function from Kendall & Gal (2017) for fine-tuning the teacher model.
    """
    if not isinstance(y_pred, dict) or 'logits' not in y_pred or 'log_variances' not in y_pred:
        raise ValueError("y_pred must be a dictionary with 'logits' and 'log_variances' keys.")

    logits = y_pred['logits']
    log_variances = y_pred['log_variances']

    # following kendall2017, this is logits + std.dev * N(0, 1)
    x_hat = logits + tf.sqrt(tf.exp(log_variances)) * tf.random.normal(shape=tf.shape(logits), dtype=tf.float32)



    bce = tf.losses.BinaryCrossentropy(from_logits=True)
    cross_entropy_loss = bce(y_true, logits)

    precision = tf.exp(-log_variances)
    adjusted_loss = (precision * cross_entropy_loss) + log_variances

    return tf.reduce_mean(adjusted_loss)


def aleatoric_loss(y_true, y_pred) -> tf.Tensor:
    """
    Aleatoric uncertainty loss function from Kendall & Gal (2017) for fine-tuning the teacher model (Eq. 12).

    https://stats.stackexchange.com/questions/573491/using-logsumexp-in-softmax
    https://lips.cs.princeton.edu/computing-log-sum-exp/

    # with logit and variance output for each batch, perform MC integration
    # both are (batch size, 1)

    # use cached MC dropout predictions
    # with mean logits and mean std. dev of each sequence, compute xhat_i,t (batch size, number of MC samples) by
    # sampling from N(0, 1) for each MC dropout sample t for each sequence i

    # with x_hat shape (i, t, c') and y_true shape (i, c), compute the loss for the whole batch
    # note: c' is the number of classes, c is the number of classes for the task
    """
    mean_logits = y_pred['mean_logits']  # Shape: (batch_size, 1)
    mean_log_variances = y_pred['mean_log_variances']  # Shape: (batch_size, 1)

    batch_size = tf.shape(mean_logits)[0]
    num_MC_samples = 20  # needs to be consistent across loss and cached mc dropout predict

    # Expand mean_logits and log_variances to include the MC sample dimension
    mean_logits_expanded = tf.tile(mean_logits, [1, num_MC_samples])  # Shape: (batch_size, num_MC_samples)
    std_dev = tf.sqrt(tf.exp(mean_log_variances))  # Standard deviation from log variance
    std_dev_expanded = tf.tile(std_dev, [1, num_MC_samples])  # Shape: (batch_size, num_MC_samples)

    # Sample from standard normal distribution
    epsilon = tf.random.normal(shape=(batch_size, 20))
    # Compute the corrupted logits (x_hat)
    x_hat = mean_logits_expanded + std_dev_expanded * epsilon  # Shape: (batch_size, num_MC_samples)
    # Compute the corrupted probabilities
    sigmoid_probs = tf.nn.sigmoid(x_hat)

    # Transform y_true into shape (batch_size, num_MC_samples)
    y_true_MC_dropout = tf.tile(tf.expand_dims(y_true, 1), tf.constant([1, num_MC_samples], dtype=tf.int32))

    # Compute the probabilities for the true labels
    probs = tf.where(y_true_MC_dropout == 1, sigmoid_probs, 1 - sigmoid_probs)  # output shape: (batch_size, num_MC_samples)

    # Compute the mean probability for each sequence
    mean_probs = tf.reduce_mean(probs, axis=1)  # output shape: (batch_size,)

    # Compute the log probabilities for the mean logits
    log_mean_probs = tf.math.log(mean_probs)  # output shape: (batch_size,)

    # Negative log likelihood loss for the whole batch
    loss = -tf.reduce_mean(log_mean_probs)  # output shape: (1,)

    return loss


# The following was adapted from https://github.com/kyle-dorman/bayesian-neural-network-blogpost/tree/master.
def bayesian_binary_crossentropy(T):
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
    try:
        logits = y_pred['logits']
        loss = tf.keras.losses.binary_crossentropy(y_true, logits, from_logits=True)
        return loss
    except KeyError:
        raise KeyError("y_pred must be a dict with key 'logits'.")


def gaussian_mle_loss(y_true, y_pred) -> tf.Tensor:
    """
    Gaussian MLE loss function from Shen et al. (2021) for fine-tuning the student model on the teacher's predictions.

    y_true = the teacher logits
    y_pred = dict with keys 'logits' and 'log_variances', the outputs of the student model heads
    # TODO: check in existing implementations if this is correct!
    """
    try:
        logits, log_variances = y_pred['logits'], y_pred['log_variances']
        # calculate the Gaussian MLE loss
        # loss = 0.5 * tf.reduce_mean(variance_i + tf.square(y_true - mu_i) / variance_i - log_variance_i
        loss = tf.reduce_mean(0.5 * tf.exp(-log_variances) * tf.norm(y_true - logits, ord='euclidean') + 0.5 * log_variances)
        # TODO: check if this corresponds to Eq. 10 in Shen et al. (2021)
        # TODO: it doesn't!
        return loss
    except KeyError:
        raise KeyError("y_pred must be a dict with keys 'logits' and 'log_variances'.")


def shen_loss(y_true, y_pred) -> tf.Tensor:
    """
    Transfer learning loss function from Shen et al. (2021) for fine-tuning the student model.
    Weight corresponds to Lambda in the paper.

    y_true = Tuple(actual ground truth - CLASS LABELS, teacher predictive sample - LOGITS)
    y_pred = dict with keys 'logits' and 'log_variances', the outputs of the student model heads
    """
    weight = tf.convert_to_tensor(1, dtype=tf.float32)

    try:
        y_true, y_teacher = y_true[0], y_true[1]

        bbc_loss = bayesian_binary_crossentropy(50)
        Lt = bbc_loss(y_true, y_pred)
        Ls = gaussian_mle_loss(y_teacher, y_pred)
        Ltotal = Ls + weight * Lt

        return Ltotal
    except KeyError:
        raise KeyError("y_true must be a dict with keys 'labels' and 'predictions'.")
