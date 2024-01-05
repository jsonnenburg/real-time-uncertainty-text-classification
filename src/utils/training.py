import tensorflow as tf
from dataclasses import dataclass


class HistorySaver(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        super(HistorySaver, self).__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.file_path, 'a') as f:
            f.write(f"Epoch {epoch + 1}: {logs}\n")


@dataclass
class BiLSTMConfig:
    embedding_dropout_rate: float
    hidden_dropout_rate: 0.1
    lstm_units_1: int
    lstm_units_2: int


@dataclass
class TextCNNConfig:
    filter_sizes: list
    num_filters: int
    dropout_rate: float
