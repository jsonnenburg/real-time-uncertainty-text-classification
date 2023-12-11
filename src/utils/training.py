import tensorflow as tf


class HistorySaver(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        super(HistorySaver, self).__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.file_path, 'a') as f:
            f.write(f"Epoch {epoch + 1}: {logs}\n")
