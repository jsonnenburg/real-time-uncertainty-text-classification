import tensorflow as tf
from src.utils.training import TextCNNConfig

# Parameters as per the paper (kim 2014)
vocab_size = 10000  # Size of the vocabulary
embedding_size = 200  # Dimensionality of the embeddings
filter_sizes = [3, 4, 5]  # Filter sizes as mentioned in the paper
num_filters = 100  # Number of filters per filter size


def create_textcnn_config(filter_sizes=None, num_filters=100, dropout_rate=0.5):
    if filter_sizes is None:
        filter_sizes = [3, 4, 5]
    config = TextCNNConfig(filter_sizes=filter_sizes, num_filters=num_filters, dropout_rate=dropout_rate)

    if not isinstance(dropout_rate, float) or not 0 <= dropout_rate < 1:
        raise ValueError("hidden_dropout_prob must be a float in the range [0, 1).")

    return config


class TextCNN(tf.keras.Model):
    def __init__(self, config, embedding_matrix, sequence_length):
        super().__init__()

        self.num_filters = config.num_filters
        self.filter_sizes = config.filter_sizes
        self.dropout_rate = config.dropout_rate

        self.embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],
                                                   output_dim=embedding_matrix.shape[1],
                                                   input_length=sequence_length,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.conv_layers = []
        for filter_size in self.filter_sizes:
            conv_layer = tf.keras.layers.Conv1D(filters=self.num_filters,
                                                kernel_size=filter_size,
                                                activation='relu')
            self.conv_layers.append(conv_layer)
        self.max_pool_layers = [tf.keras.layers.GlobalMaxPooling1D() for _ in self.filter_sizes]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.classifier = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.),
            name="classifier",
            trainable=True
        )

    def call(self, inputs, training=False, mask=None):
        """
        Returns logits of shape [batch_size, num_classes].
        """
        x = self.embedding(inputs)
        pooled_outputs = []
        for i, conv_layer in enumerate(self.conv_layers):
            conv = conv_layer(x)
            pooled = self.max_pool_layers[i](conv)
            pooled_outputs.append(pooled)
        total_filters = len(self.conv_layers) * self.num_filters
        x = tf.concat(pooled_outputs, axis=1)
        x = tf.reshape(x, (-1, total_filters))
        x = self.dropout(x, training=training)
        x = self.classifier(x)
        return x

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
        config = super(TextCNN, self).get_config()
        config.update({
            "filter_sizes": self.filter_sizes,
            "num_filters": self.num_filters,
            "dropout_rate": self.dropout_rate,
        })
        return config
