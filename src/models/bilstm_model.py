import tensorflow as tf
from src.utils.training import BiLSTMConfig

# TODO: add weight decay ? -> Adam optimizer


def create_bilstm_config(embedding_dropout_rate=0.5, hidden_dropout_rate=0.5, lstm_units_1=128, lstm_units_2=64):
    config = BiLSTMConfig(embedding_dropout_rate=embedding_dropout_rate, hidden_dropout_rate=hidden_dropout_rate,
                        lstm_units_1=lstm_units_1, lstm_units_2=lstm_units_2)

    if not isinstance(hidden_dropout_rate, float) or not 0 <= hidden_dropout_rate < 1 or \
            not isinstance(embedding_dropout_rate, float) or not 0 <= embedding_dropout_rate < 1:
        raise ValueError("dropout probabilities must be floats in the range [0, 1).")

    return config


class BiLSTM(tf.keras.Model):
    """
    Bidirectional LSTM model for sequence classification.

    :param embedding_matrix: Pre-trained embedding matrix for initializing the embedding layer.
    :param sequence_length: Length of the input sequences.
    :param lstm_units: Number of units in each LSTM layer.
    :param num_classes: Number of output classes for classification.
    :param dropout_rate: Dropout rate for regularization.
    """

    def __init__(self, config, embedding_matrix, sequence_length: int = 48):
        super().__init__()

        self.embedding_dropout_rate = config.embedding_dropout_rate
        self.hidden_dropout_rate = config.hidden_dropout_rate
        self.lstm_units_1 = config.lstm_units_1
        self.lstm_units_2 = config.lstm_units_2

        self.embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],
                                                   output_dim=embedding_matrix.shape[1],
                                                   input_length=sequence_length,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.embedding_dropout = tf.keras.layers.Dropout(self.embedding_dropout_rate)
        self.bilstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units_1, return_sequences=True))
        self.hidden_dropout_1 = tf.keras.layers.Dropout(self.hidden_dropout_rate)
        self.bilstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units_2, return_sequences=False))
        self.hidden_dropout_2 = tf.keras.layers.Dropout(self.hidden_dropout_rate)
        self.classifier = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.),
            name="classifier",
            trainable=True
        )

    def call(self, inputs, training=False, mask=None):
        x = self.embedding(inputs)
        x = self.embedding_dropout(x, training=training)
        x = self.bilstm_1(x)
        x = self.hidden_dropout_1(x, training=training)
        x = self.bilstm_2(x)
        x = self.hidden_dropout_2(x, training=training)
        return self.classifier(x)

    def mc_dropout_sample(self, inputs, n=50):
        """
        Performs MC dropout sampling.

        :param inputs: Input sample.
        :param n: Number of forward passes to average over.
        :return: Dictionary containing logits and probabilities, mean and variance of predictions.
        """
        all_logits = []
        all_probs = []

        for i in range(n):
            logits = self(inputs, training=True)
            probs = tf.nn.sigmoid(logits)
            all_logits.append(logits)
            all_probs.append(probs)

        all_logits = tf.stack(all_logits, axis=1)
        all_probs = tf.stack(all_probs, axis=1)
        mean_predictions = tf.reduce_mean(all_logits, axis=1)
        var_predictions = tf.math.reduce_variance(all_logits, axis=1)

        return {'logit_samples': all_logits,
                'prob_samples': all_probs,
                'mean_logits': mean_predictions,
                'var_logits': var_predictions,
                }

    def get_config(self):
        config = super(BiLSTM, self).get_config()
        config.update({
            "embedding_dropout_rate": self.embedding_dropout_rate,
            "hidden_dropout_rate": self.hidden_dropout_rate,
            "lstm_units_1": self.lstm_units_1,
            "lstm_units_2": self.lstm_units_2,
            "sequence_length": self.sequence_length,
        })
        return config
