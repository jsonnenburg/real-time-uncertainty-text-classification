import tensorflow as tf

# Parameters as per the paper (kim 2014)
vocab_size = 10000  # Size of the vocabulary
embedding_size = 300  # Dimensionality of the embeddings
filter_sizes = [3, 4, 5]  # Filter sizes as mentioned in the paper
num_filters = 100  # Number of filters per filter size


class TextCNN(tf.keras.Model):
    def __init__(self, embedding_matrix, sequence_length, filter_sizes, num_filters, num_classes, dropout_rate):

        super().__init__()

        self.embedding = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],
                                                   output_dim=embedding_matrix.shape[1],
                                                   input_length=sequence_length,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        self.conv_layers = []
        for filter_size in filter_sizes:
            conv_layer = tf.keras.layers.Conv1D(filters=num_filters,
                                                kernel_size=filter_size,
                                                activation='relu')
            self.conv_layers.append(conv_layer)
        self.max_pool_layers = [tf.keras.layers.GlobalMaxPooling1D() for _ in filter_sizes]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        x = self.embedding(x)
        pooled_outputs = []
        for i, conv_layer in enumerate(self.conv_layers):
            conv = conv_layer(x)
            pooled = self.max_pool_layers[i](conv)
            pooled_outputs.append(pooled)
        total_filters = len(self.conv_layers) * num_filters
        x = tf.concat(pooled_outputs, axis=1)
        x = tf.reshape(x, (-1, total_filters))
        x = self.dropout(x, training=training)
        x = self.classifier(x)
        return x
