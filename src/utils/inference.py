import tensorflow as tf


def mc_dropout_predict(model, inputs, n=20, seed_list=None):
    """
    Computes the mean and variance of the predictions of a model with MC dropout enabled over N samples.
    """
    all_logits = []

    if seed_list is None:
        seed_list = range(n)

    for i in range(n):
        tf.random.set_seed(seed_list[i])
        outputs = model(inputs, training=True)
        logits = outputs['logits']
        all_logits.append(logits)

    all_logits = tf.stack(all_logits, axis=0)
    mean_predictions = tf.reduce_mean(all_logits, axis=0)
    var_predictions = tf.math.reduce_variance(all_logits, axis=0)

    return all_logits, mean_predictions, var_predictions
