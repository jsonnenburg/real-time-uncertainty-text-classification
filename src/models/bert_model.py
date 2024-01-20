from __future__ import annotations

from typing import Optional

from transformers import BertConfig, TFBertModel
import tensorflow as tf
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput

from src.utils.inference import compute_total_uncertainty


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
            # activation='linear',
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
            y_pred_mc = self.minimal_cached_mc_dropout_predict(x, n=20)
            if not isinstance(y_pred, CustomTFSequenceClassifierOutput):
                raise TypeError("The output of the model is not CustomTFSequenceClassifierOutput.")
            if self.custom_loss_fn is not None:
                loss = self.custom_loss_fn(
                    y,
                    {
                        'logits': y_pred.logits,
                        'log_variances': y_pred.log_variances,
                        'mean_logits': y_pred_mc.mean_logits,
                        'mean_log_variances': y_pred_mc.mean_log_variances
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
        y_pred_mc = self.minimal_cached_mc_dropout_predict(x, n=20)
        if not isinstance(y_pred, CustomTFSequenceClassifierOutput):
            raise TypeError("The output of the model is not CustomTFSequenceClassifierOutput.")
        if self.custom_loss_fn is not None:
            loss = self.custom_loss_fn(
                y,
                {
                    'logits': y_pred.logits,
                    'log_variances': y_pred.log_variances,
                    'mean_logits': y_pred_mc.mean_logits,
                    'mean_log_variances': y_pred_mc.mean_log_variances
                }
            )
        else:
            raise ValueError("No custom loss function provided!")

        self.compiled_metrics.update_state(y, y_pred.probs)

        return {m.name: m.result() for m in self.metrics}

    def cached_mc_dropout_predict(self, inputs, dropout_rate: Optional[float] = None, n=20) -> dict:
        bert_outputs = self.bert(inputs, training=False)
        pooled_output = bert_outputs.pooler_output

        if dropout_rate is not None:
            self.dropout.rate = dropout_rate

        all_logits = []
        all_probs = []
        all_log_variances = []
        for i in range(n):
            tf.random.set_seed(range(n)[i])
            dropout_output = self.dropout(pooled_output, training=True)
            logits = self.classifier(dropout_output)
            probs = tf.nn.sigmoid(logits)
            log_variances = self.log_variance_predictor(dropout_output)
            all_logits.append(logits)
            all_probs.append(probs)
            all_log_variances.append(log_variances)

        all_logits = tf.stack(all_logits, axis=0)
        all_probs = tf.stack(all_probs, axis=0)
        all_log_variances = tf.stack(all_log_variances, axis=0)
        mean_predictions = tf.reduce_mean(all_logits, axis=0)
        var_predictions = tf.math.reduce_variance(all_logits, axis=0)

        epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty = compute_total_uncertainty(all_logits,
                                                                                                    all_log_variances)
        mean_variances = aleatoric_uncertainty

        return {'logits': all_logits,
                'probs': all_probs,
                'log_variances': all_log_variances,
                'mean_predictions': mean_predictions,
                'mean_variances': mean_variances,
                'var_predictions': var_predictions,
                'total_uncertainty': total_uncertainty,
                }

    def minimal_cached_mc_dropout_predict(self, inputs, dropout_rate: Optional[float] = None, n=20) -> dict:
        bert_outputs = self.bert(inputs, training=False)
        pooled_output = bert_outputs.pooler_output

        if dropout_rate is not None:
            self.dropout.rate = dropout_rate

        all_logits = []
        all_probs = []
        all_log_variances = []
        for i in range(n):
            tf.random.set_seed(range(n)[i])
            dropout_output = self.dropout(pooled_output, training=True)
            logits = self.classifier(dropout_output)
            probs = tf.nn.sigmoid(logits)
            log_variances = self.log_variance_predictor(dropout_output)
            all_logits.append(logits)
            all_log_variances.append(log_variances)

        all_logits = tf.stack(all_logits, axis=0)
        all_probs = tf.stack(all_probs, axis=0)
        all_log_variances = tf.stack(all_log_variances, axis=0)
        mean_predictions = tf.reduce_mean(all_logits, axis=0)
        mean_log_variances = tf.math.reduce_variance(all_log_variances, axis=0)

        return {'probs': all_probs,
                'mean_logits': mean_predictions,
                'mean_log_variances': mean_log_variances
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


class AleatoricMCDropoutBERTStudent(tf.keras.Model):
    def __init__(self, config, custom_loss_fn=None):
        super(AleatoricMCDropoutBERTStudent, self).__init__()
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
            activation='linear',
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

    def cached_mc_dropout_predict(self, inputs, n=20) -> dict:
        bert_outputs = self.bert(inputs, training=False)
        pooled_output = bert_outputs.pooler_output

        all_logits = []
        all_probs = []
        all_log_variances = []
        for i in range(n):
            tf.random.set_seed(range(n)[i])
            dropout_output = self.dropout(pooled_output, training=True)
            logits = self.classifier(dropout_output)
            probs = tf.nn.sigmoid(logits)
            log_variances = self.log_variance_predictor(dropout_output)
            all_logits.append(logits)
            all_probs.append(probs)
            all_log_variances.append(log_variances)

        all_logits = tf.stack(all_logits, axis=0)
        all_probs = tf.stack(all_probs, axis=0)
        all_log_variances = tf.stack(all_log_variances, axis=0)
        mean_predictions = tf.reduce_mean(all_logits, axis=0)
        var_predictions = tf.math.reduce_variance(all_logits, axis=0)

        epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty = compute_total_uncertainty(all_logits,
                                                                                                    all_log_variances)
        mean_variances = aleatoric_uncertainty

        return {'logits': all_logits,
                'probs': all_probs,
                'log_variances': all_log_variances,
                'mean_predictions': mean_predictions,
                'mean_variances': mean_variances,
                'var_predictions': var_predictions,
                'total_uncertainty': total_uncertainty,
                }

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if not isinstance(y_pred, CustomTFSequenceClassifierOutput):
                raise TypeError("The output of the model is not CustomTFSequenceClassifierOutput.")
            if self.custom_loss_fn is not None:
                loss = self.custom_loss_fn(y, {'logits': y_pred.logits, 'log_variances': y_pred.log_variances})
            else:
                raise ValueError("No custom loss function provided!")

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred.probs)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        y_pred = self.call(x, training=False)
        if not isinstance(y_pred, CustomTFSequenceClassifierOutput):
            raise TypeError("The output of the model is not CustomTFSequenceClassifierOutput.")
        if self.custom_loss_fn is not None:
            loss = self.custom_loss_fn(y, {'logits': y_pred.logits, 'log_variances': y_pred.log_variances})
        else:
            raise ValueError("No custom loss function provided!")

        self.compiled_metrics.update_state(y, y_pred.probs)

        return {m.name: m.result() for m in self.metrics}


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
