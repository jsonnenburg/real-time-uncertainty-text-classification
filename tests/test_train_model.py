import unittest
from unittest.mock import patch, MagicMock

import numpy as np

import tensorflow as tf

from src.models.bert_model import CustomTFSequenceClassifierOutput
from src.training.train_bert_teacher import train_model


class TestTrainModel(unittest.TestCase):

    @patch('src.training.train_bert_teacher.MCDropoutBERT')
    @patch('src.training.train_bert_teacher.bert_preprocess')
    @patch('src.training.train_bert_teacher.get_tf_dataset')
    def test_train_model(self, mock_get_tf_dataset, mock_bert_preprocess, mock_MCDropoutBERT):
        # Mock the dependencies
        mock_model = MagicMock()
        mock_MCDropoutBERT.from_pretrained.return_value = mock_model

        # Setup test data
        mock_dataset = MagicMock()
        mock_dataset.train = MagicMock()
        mock_dataset.val = MagicMock()
        mock_dataset.test = MagicMock()

        mock_bert_preprocess.return_value = mock_dataset

        # Mock TensorFlow dataset
        mock_tf_dataset = tf.data.Dataset.from_tensor_slices((np.random.rand(10, 2), np.random.randint(2, size=10)))
        mock_get_tf_dataset.return_value = mock_tf_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        mock_mean_predictions = np.random.rand(10, 2)  # Adjust the shape as needed
        mock_model.predict.return_value = CustomTFSequenceClassifierOutput(logits=mock_mean_predictions,
                                                                           labels=np.random.randint(2, size=10))

        with patch('src.training.train_bert_teacher.mc_dropout_predict') as mock_mc_dropout_predict:
            mock_mc_dropout_predict.return_value = (
            mock_mean_predictions, None)  # Replace None with variance predictions if needed

            # Define a dummy config
            dummy_config = MagicMock()

            # Call the function with test parameters
            train_model(config=dummy_config, dataset=mock_dataset, batch_size=32, learning_rate=0.001, epochs=3)

        # Assert model was compiled and fit was called
        mock_model.compile.assert_called_once()
        mock_model.fit.assert_called_once()

        # Additional assertions can be added here based on expected behavior

    # Additional test cases can be added here for different scenarios


if __name__ == '__main__':
    unittest.main()
