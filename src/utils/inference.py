import tensorflow as tf


def mc_dropout_predict(model, inputs, n=20):
    """Computes the mean and variance of the predictions of a model with MC Dropout enabled over N samples.
    """
    predictions = [model(inputs, training=True) for _ in range(n)]
    predictions = tf.stack(predictions)
    mean_predictions = tf.reduce_mean(predictions, axis=0)
    var_predictions = tf.math.reduce_variance(predictions, axis=0)
    return mean_predictions, var_predictions
