import tensorflow as tf


def compute_total_uncertainty(all_logits, all_log_variances):
    # squared logits, summed over MC samples
    mean_of_squares = tf.reduce_mean(tf.square(all_logits), axis=0)
    # squared sum of logits, summed over MC samples
    square_of_means = tf.square(tf.reduce_mean(all_logits, axis=0))
    # variance of logits, averaged over MC samples
    mean_variances = tf.reduce_mean(tf.exp(all_log_variances), axis=0)

    epistemic_uncertainty = mean_of_squares - square_of_means
    aleatoric_uncertainty = mean_variances
    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

    return epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty


def mc_dropout_predict(model, inputs, n=20, seed_list=None):
    """
    Computes the mean and variance of the predictions of a model with MC dropout enabled over N samples.
    """
    all_logits = []
    all_log_variances = []

    if seed_list is None:
        seed_list = range(n)

    for i in range(n):
        tf.random.set_seed(seed_list[i])
        outputs = model(inputs, training=True)
        logits = outputs.logits
        log_variances = outputs.log_variances
        all_logits.append(logits)
        all_log_variances.append(log_variances)

    all_logits = tf.stack(all_logits, axis=0)
    all_log_variances = tf.stack(all_log_variances, axis=0)
    mean_predictions = tf.reduce_mean(all_logits, axis=0)
    var_predictions = tf.math.reduce_variance(all_logits, axis=0)

    epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty = compute_total_uncertainty(all_logits, all_log_variances)
    mean_variances = aleatoric_uncertainty

    return all_logits, mean_variances, mean_predictions, var_predictions, total_uncertainty
