from src.training.train_bert_teacher import compute_mc_dropout_metrics, compute_metrics
import tensorflow as tf
import unittest


def simulate_bert_output(batch_size, num_classes):
    logits = tf.random.normal((batch_size, num_classes))
    log_variances = tf.random.normal((batch_size, num_classes))
    return logits, log_variances


def simulate_mc_dropout_bert_output(batch_size, num_classes, num_samples):
    all_logits = tf.random.normal((num_samples, batch_size, num_classes))
    mean_logits = tf.reduce_mean(all_logits, axis=0)
    return mean_logits


class TestComputeMetrics(unittest.TestCase):

    def test_compute_metrics(self):
        # Simulate data
        batch_size = 32
        num_classes = 2
        logits, log_variances = simulate_bert_output(batch_size, num_classes)

        # Simulate labels
        labels = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)

        # Create a mock prediction object
        pred = {
            "label_ids": labels,
            "predictions": logits
        }

        # Compute metrics
        metrics = compute_metrics(pred)

        # Here you can add assertions to check if metrics are within expected ranges
        self.assertIn("accuracy_score", metrics)
        self.assertIn("precision_score", metrics)
        # Add other assertions as needed


class TestComputeMCDropoutMetrics(unittest.TestCase):

    def test_compute_mc_dropout_metrics(self):
        # Simulate data
        batch_size = 32
        num_classes = 2
        num_samples = 20
        mean_logits = simulate_mc_dropout_bert_output(batch_size, num_classes, num_samples)

        # Simulate labels
        labels = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)

        # Compute metrics
        metrics = compute_mc_dropout_metrics(labels, mean_logits)

        # Here you can add assertions to check if metrics are within expected ranges
        self.assertIn("accuracy_score", metrics)
        self.assertIn("precision_score", metrics)
        # Add other assertions as needed


if __name__ == '__main__':
    unittest.main()
