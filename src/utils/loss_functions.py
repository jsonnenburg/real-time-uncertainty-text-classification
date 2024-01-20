import tensorflow as tf


def null_loss(y_true, y_pred) -> tf.Tensor:
    """
    Null loss function (dummy loss) for fine-tuning the custom aleatoric uncertainty BERT models.
    """
    return tf.zeros_like(y_true)


def archived_aleatoric_loss(y_true, y_pred) -> tf.Tensor:
    """
    Aleatoric uncertainty loss function from Kendall & Gal (2017) for fine-tuning the teacher model.
    TODO: turn this into eq. 12 from Kendall & Gal (2017)
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
    """
    # with logit and variance output for each batch, perform MC integration
    # both are (batch size, 1)

    # use cached MC dropout predictions TODO: move this to the train_step and test_step functions in the model
    # with mean logits and mean std. dev of each sequence, compute xhat_i,t (batch size, number of MC samples) by
    # sampling from N(0, 1) for each MC dropout sample t for each sequence i

    # with x_hat shape (i, t, c') and y_true shape (i, c), compute the loss for the whole batch
    # note: c' is the number of classes, c is the number of classes for the task

    mean_logits = y_pred['mean_logits']
    mean_log_variances = y_pred['mean_log_variances']

    batch_size = tf.shape(mean_logits)[0]
    num_MC_samples = 20  # needs to be consistent across loss and cached mc dropout predict

    # Expand mean_logits and log_variances to include the MC sample dimension
    mean_logits_expanded = tf.expand_dims(mean_logits, 1)  # Shape: (batch_size, 1)
    std_dev = tf.sqrt(tf.exp(mean_log_variances))  # Standard deviation from log variance
    std_dev_expanded = tf.expand_dims(std_dev, 1)  # Shape: (batch_size, 1)

    # Sample from standard normal distribution
    epsilon = tf.random.normal(shape=(batch_size, 20))
    # Compute the corrupted logits (x_hat)
    x_hat = mean_logits_expanded + std_dev_expanded * epsilon  # Shape: (batch_size, num_MC_samples)
    # Compute the corrupted probabilities
    sigmoid_probs = tf.nn.sigmoid(x_hat)

    # transform y_true into shape (batch_size, MC_samples)
    y_true_MC_dropout = tf.tile(y_true, [1, num_MC_samples])

    # Compute the probabilities for the true labels
    probs = tf.where(y_true_MC_dropout == 1,
                         sigmoid_probs,
                         1 - sigmoid_probs)  # Shape: (batch_size, num_MC_samples)

    # Compute the mean probability for each sequence
    mean_probs = tf.reduce_mean(probs, axis=1)  # Shape: (batch_size,)

    # Compute the log probabilities for the mean logits
    log_mean_probs = tf.math.log(mean_probs)  # Shape: (batch_size,)

    # Negative log likelihood loss for the whole batch
    loss = tf.reduce_mean(log_mean_probs) # Shape: (1,)

    return loss


def bce_loss(y_true, y_pred) -> tf.Tensor:
    """
    Binary cross entropy loss function for fine-tuning the student model.
    """
    try:
        logits = y_pred['logits']
        bce = tf.losses.BinaryCrossentropy(from_logits=True)
        loss = bce(y_true, logits)
        return loss
    except KeyError:
        raise KeyError("y_pred must be a dict with key 'logits'.")


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

    y_true = Tuple(actual ground truth - CLASS LABELS, teacher predictive sample - LOGITS)
    y_pred = dict with keys 'logits' and 'log_variances', the outputs of the student model heads
    """
    weight = tf.convert_to_tensor(1, dtype=tf.float32)

    try:
        y_true, y_teacher = y_true[0], y_true[1]

        Lt = aleatoric_loss(y_true, y_pred)
        Ls = gaussian_mle_loss(y_teacher, y_pred)
        Ltotal = Ls + weight * Lt

        return Ltotal
    except KeyError:
        raise KeyError("y_true must be a dict with keys 'labels' and 'predictions'.")
