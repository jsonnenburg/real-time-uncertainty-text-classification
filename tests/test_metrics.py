from src.experiments.uncertainty_distillation.train_bert_teacher import compute_mc_dropout_metrics, compute_metrics
import unittest
import numpy as np
import tensorflow as tf
from src.utils.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, nll_score, brier_score,
    pred_entropy_score, ece_score
)


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
        batch_size = 32
        num_classes = 2
        logits, log_variances = simulate_bert_output(batch_size, num_classes)

        labels = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)

        pred = {
            "label_ids": labels,
            "predictions": logits
        }

        metrics = compute_metrics(pred)

        self.assertIn("accuracy_score", metrics)
        self.assertIn("precision_score", metrics)


class TestComputeMCDropoutMetrics(unittest.TestCase):

    def test_compute_mc_dropout_metrics(self):
        batch_size = 32
        num_classes = 2
        num_samples = 20
        mean_logits = simulate_mc_dropout_bert_output(batch_size, num_classes, num_samples)

        labels = tf.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tf.int32)

        metrics = compute_mc_dropout_metrics(labels, mean_logits)

        self.assertIn("accuracy_score", metrics)
        self.assertIn("precision_score", metrics)


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.num_classes = 2
        self.y_true = np.random.randint(0, self.num_classes, self.batch_size)
        self.y_pred = np.random.randint(0, self.num_classes, self.batch_size)
        self.y_prob = np.random.rand(self.batch_size, self.num_classes)

    def test_accuracy_score(self):
        accuracy = accuracy_score(self.y_true, self.y_pred)
        self.assertTrue(0 <= accuracy <= 1)

    def test_precision_score(self):
        precision = precision_score(self.y_true, self.y_pred)
        self.assertTrue(0 <= precision <= 1)

    def test_recall_score(self):
        recall = recall_score(self.y_true, self.y_pred)
        self.assertTrue(0 <= recall <= 1)

    def test_f1_score(self):
        f1 = f1_score(self.y_true, self.y_pred)
        self.assertTrue(0 <= f1 <= 1)

    def test_nll_score(self):
        nll = nll_score(self.y_true, self.y_prob)
        self.assertTrue(nll >= 0)

    def test_brier_score(self):
        brier = brier_score(self.y_true, self.y_prob[:, 1])
        self.assertTrue(0 <= brier <= 1)

    def test_pred_entropy_score(self):
        entropy = pred_entropy_score(self.y_prob)
        self.assertTrue((entropy >= 0).all() and (entropy <= 1).all())

    def test_ece_score(self):
        ece = ece_score(self.y_true, self.y_pred, self.y_prob[:, 1])
        self.assertTrue(0 <= ece <= 1)


if __name__ == '__main__':
    unittest.main()
