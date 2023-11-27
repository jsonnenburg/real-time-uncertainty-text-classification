import tensorflow as tf


class BiLSTM(tf.keras.Model):
    """Bidirectional LSTM model for sequence classification.
    TODO: use GloVe embeddings?
    TODO: adapt this to the actual model architecture decided on

    :param embedding_matrix: Pre-trained embedding matrix for initializing the embedding layer.
    :param lstm_units: Number of units in each LSTM layer.
    :param num_classes: Number of output classes for classification.
    :param dropout_rate: Dropout rate for regularization.
    """

    def __init__(self, embedding_matrix, sequence_length, lstm_units, num_classes, dropout_rate):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],
                                                   output_dim=embedding_matrix.shape[1],
                                                   input_length=sequence_length,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.embedding_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bilstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))
        self.hidden_dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.bilstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=False))
        self.hidden_dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        x = self.embedding(x)
        x = self.embedding_dropout(x, training=training)
        x = self.bilstm_1(x)
        x = self.hidden_dropout_1(x, training=training)
        x = self.bilstm_2(x)
        x = self.hidden_dropout_2(x, training=training)
        return self.classifier(x)


# we can additionally define the student architecture here, which is a similar BiLSTM but with two outputs,
# one for the mean (classifier) and one for the log variance (regressor)
# this model can then be initialized with the fine-tuned weights of the standard BiLSTM model

class BiLSTMStudent(BiLSTM):
    """Extension of the BiLSTMModel with an additional output for regression.

    This model includes two heads: one for classification (mean prediction) and
    one for regression (log variance prediction).

    :param embedding_matrix: Pre-trained embedding matrix for initializing the embedding layer.
    :param lstm_units: Number of units in each LSTM layer.
    :param num_classes: Number of output classes for classification.
    :param dropout_rate: Dropout rate for regularization.
    """

    def __init__(self, embedding_matrix, lstm_units, num_classes, dropout_rate):
        super().__init__(embedding_matrix, lstm_units, num_classes, dropout_rate)

        self.regressor = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        """
        Set training to True to for MC Dropout.
        """
        x = super().call(inputs, training=training)
        return self.classifier(x), self.regressor(x)
