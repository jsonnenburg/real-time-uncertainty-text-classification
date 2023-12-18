import tensorflow as tf


def mc_dropout_predict(model, inputs, n=20):
    """
    Computes the mean and variance of the predictions of a model with MC dropout enabled over N samples.
    """
    all_logits = []

    for _ in range(n):
        outputs = model(inputs, training=True)
        logits = outputs['logits']
        all_logits.append(logits)

    all_logits = tf.stack(all_logits, axis=0)
    mean_predictions = tf.reduce_mean(all_logits, axis=0)
    var_predictions = tf.math.reduce_variance(all_logits, axis=0)

    return all_logits, mean_predictions, var_predictions
