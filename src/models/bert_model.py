from __future__ import annotations

import random

from transformers import BertConfig, TFBertModel
import tensorflow as tf
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput


def create_bert_config(hidden_dropout_prob, attention_probs_dropout_prob, classifier_dropout):
    config = BertConfig()
    config.hidden_dropout_prob = hidden_dropout_prob
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.classifier_dropout = classifier_dropout

    if not isinstance(hidden_dropout_prob, float) or not 0 <= hidden_dropout_prob < 1:
        raise ValueError("hidden_dropout_prob must be a float in the range [0, 1).")

    return config


class CustomTFSequenceClassifierOutput(TFSequenceClassifierOutput):
    def __init__(self, labels=None, loss=None, logits=None, probs=None, hidden_states=None, attentions=None, log_variances=None):
        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
        self.labels = labels
        self.probs = probs
        self.log_variances = log_variances


class AleatoricMCDropoutBERT(tf.keras.Model):
    def __init__(self, config, custom_loss_fn=None):
        super(AleatoricMCDropoutBERT, self).__init__()
        self.bert = TFBertModel.from_pretrained(
            'bert-base-uncased',
            config=config
        )
        self.dropout = tf.keras.layers.Dropout(
            rate=config.classifier_dropout
        )
        self.classifier = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="classifier",
            trainable=True
        )
        self.log_variance_predictor = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=config.initializer_range),
            name="log_variance",
            trainable=True
        )

        if custom_loss_fn:
            self.custom_loss_fn = custom_loss_fn
        else:
            self.custom_loss_fn = tf.keras.losses.BinaryCrossentropy()

    def call(self, inputs, training=False, mask=None):
        bert_outputs = self.bert(inputs, training=training)
        pooled_output = bert_outputs.pooler_output
        pooled_output = self.dropout(pooled_output, training=training)

        logits = self.classifier(pooled_output)
        probs = tf.nn.sigmoid(logits)
        log_variances = self.log_variance_predictor(pooled_output)

        return CustomTFSequenceClassifierOutput(
            logits=logits,
            probs=probs,
            log_variances=log_variances,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions
        )

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if self.custom_loss_fn is not None:
                loss = self.custom_loss_fn(
                    y,
                    {
                        'logits': y_pred.logits,
                        'log_variances': y_pred.log_variances
                    }
                )
            else:
                raise ValueError("No custom loss function provided!")

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred.probs)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        y_pred = self(x, training=False)
        if self.custom_loss_fn is not None:
            loss = self.custom_loss_fn(
                y,
                {
                    'logits': y_pred.logits,
                    'log_variances': y_pred.log_variances
                }
            )
        else:
            raise ValueError("No custom loss function provided!")

        self.compiled_metrics.update_state(y, y_pred.probs)

        return {m.name: m.result() for m in self.metrics}

    def monte_carlo_sample(self, inputs, n=50) -> dict:
        """
        Performs Monte Carlo sampling over the logit space.
        """
        y_pred = self(inputs, training=False)
        logits = y_pred.logits
        log_variance = y_pred.log_variances
        variance = tf.exp(log_variance)
        std = tf.sqrt(variance)

        # sample n times from normal(0, std)
        rand_seed = random.randint(0, 2 ** 32 - 1)
        tf.random.set_seed(rand_seed)
        std_expanded = tf.expand_dims(std, -1)
        std_samples = tf.random.normal(shape=(tf.shape(logits)[0], n, 1), stddev=std_expanded)  # (batch_size, n, 1)
        # Adjust logits to have shape (batch_size, 1, 1) and then tile to match std_samples shape
        logits_expanded = tf.expand_dims(logits, 1)  # (batch_size, 1, 1)
        logits_tiled = tf.tile(logits_expanded, tf.constant([1, n, 1]))  # (batch_size, n, 1)

        # Add the tiled logits and the standard deviation samples
        logits_distorted = logits_tiled + std_samples  # (batch_size, n, 1)
        probs_distorted = tf.nn.sigmoid(logits_distorted)  # (batch_size, n, 1)

        mean_logits = tf.reduce_mean(logits_distorted, axis=1)  # (batch_size, 1)
        mean_probs = tf.nn.sigmoid(mean_logits)

        return {'logit_samples': logits_distorted,
                'prob_samples': probs_distorted,
                'mean_logits': mean_logits,
                'mean_probs': mean_probs,
                'logits': logits,
                'log_variances': log_variance,
                }

    def mc_dropout_sample(self, inputs, n=50) -> dict:
        """
        Performs MC dropout sampling.
        """
        all_logits = []
        all_log_variances = []
        for i in range(n):
            rand_seed = random.randint(0, 2 ** 32 - 1)
            tf.random.set_seed(rand_seed)
            outputs = self(inputs, training=True)
            all_logits.append(outputs.logits)
            all_log_variances.append(outputs.log_variances)
        all_logits = tf.stack(all_logits, axis=1)
        all_probs = tf.nn.sigmoid(all_logits)
        all_log_variances = tf.stack(all_log_variances, axis=1)
        mean_logits = tf.reduce_mean(all_logits, axis=1)
        mean_log_variances = tf.reduce_mean(all_log_variances, axis=1)

        return {'logit_samples': all_logits,
                'prob_samples': all_probs,
                'log_variance_samples': all_log_variances,
                'mean_logits': mean_logits,
                'mean_log_variances': mean_log_variances,
                }

    def get_config(self):
        config = {
            'bert_config': self.bert.config.to_dict(),
            'custom_loss_fn_name': self.custom_loss_fn.__name__ if self.custom_loss_fn else None,
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        bert_config = BertConfig.from_dict(config['bert_config'])

        custom_loss_fn = None
        if config['custom_loss_fn_name']:
            if config['custom_loss_fn_name'] in globals():
                custom_loss_fn = globals()[config['custom_loss_fn_name']]
            else:
                raise ValueError(f"Unknown custom loss function: {config['custom_loss_fn_name']}")

        new_model = cls(bert_config, custom_loss_fn=custom_loss_fn)
        return new_model
