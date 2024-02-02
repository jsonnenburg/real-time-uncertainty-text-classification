from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf

# remove stopwords
from nltk.corpus import stopwords
from tensorflow import Tensor

stopwords_set = set(stopwords.words("english"))
other_exclusions = {"#ff", "ff", "rt"}
stopwords_set.update(other_exclusions)


def remove_stopwords(text: str) -> str:
    return " ".join(word for word in text.split() if word not in stopwords_set)


def pad_sequences(sequences, max_length: int = 48) -> tf.Tensor:
    """
    Pads the sequences to a maximum length.
    :param sequences: The sequences.
    :param max_length: The maximum length.
    :return: The padded sequences.
    """
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)


def get_embedding_matrix(word_index: Dict, glove_embeddings: Dict, glove_vector_size: int) -> Tensor:
    embedding_matrix = np.zeros((len(word_index) + 1, glove_vector_size))
    for word, i in word_index.items():
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = np.array(embedding_vector, dtype=np.float32)
    return tf.convert_to_tensor(embedding_matrix, dtype=tf.float32)


def load_glove_embeddings(glove_file: str) -> Dict:
    """
    Loads the GloVe embeddings.
    :param glove_file: The GloVe file.
    :return: The GloVe embeddings.
    """
    embeddings_index = {}
    with open(glove_file, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = tf.constant(values[1:], dtype=tf.float32)
            embeddings_index[word] = coefs
    return embeddings_index
