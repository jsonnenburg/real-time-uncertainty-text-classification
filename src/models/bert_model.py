from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from keras.src.callbacks import TensorBoard
from tensorflow import Tensor
from transformers import TFBertForSequenceClassification, TFBertMainLayer, BertConfig, \
    TFAutoModelForSequenceClassification, TFPreTrainedModel, TFBertModel
import tensorflow as tf
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput
from transformers.modeling_tf_utils import get_initializer, TFModelInputType

from src.utils.loss_functions import shen_loss, null_loss
from src.utils.training import HistorySaver


class CustomTFSequenceClassifierOutput(TFSequenceClassifierOutput):
    def __init__(self, labels=None, loss=None, logits=None, probs=None, hidden_states=None, attentions=None, log_variances=None):
        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
        self.labels = labels
        self.probs = probs
        self.log_variances = log_variances


class CustomMCDropoutBERT(TFBertForSequenceClassification):
    def __init__(self, config, custom_loss_fn=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.bert = TFBertMainLayer(config, name="bert")
        # replace classifier with two separate heads
        self.classifier = tf.keras.layers.Dense(
            units=2,  # binary classifier TODO: maybe 768 instead?
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        self.log_variance_predictor = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=get_initializer(config.initializer_range),
            name="log_variance",
        )
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

        if custom_loss_fn:
            self.custom_loss_fn = custom_loss_fn
        else:
            self.custom_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(
            self,
            input_ids: TFModelInputType | None = None,
            attention_mask: np.ndarray | tf.Tensor | None = None,
            token_type_ids: np.ndarray | tf.Tensor | None = None,
            position_ids: np.ndarray | tf.Tensor | None = None,
            head_mask: np.ndarray | tf.Tensor | None = None,
            inputs_embeds: np.ndarray | tf.Tensor | None = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = True,  # None
            labels: np.ndarray | tf.Tensor | None = None,
            training: Optional[bool] = False,
    ) -> CustomTFSequenceClassifierOutput:
        outputs = super().call(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            False,  # want the super to return everything!
            labels,
            training,
        )

        logits = outputs.logits

        pooled_output = self.dropout(logits, training=True)  # Apply dropout to logits if that's the intention

        logits = self.classifier(pooled_output)
        log_variances: Tensor = self.log_variance_predictor(pooled_output)

        loss = None
        if labels is not None:
            loss = self.custom_loss_fn(labels, logits)

        # if not return_dict:
        #    return (loss, logits) + outputs[2:], log_variances

        return CustomTFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            log_variances=log_variances,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


class ArchivedAleatoricMCDropoutBERT(TFAutoModelForSequenceClassification, TFPreTrainedModel):
    def __init__(self, config: BertConfig):
        super(AleatoricMCDropoutBERT, self).__init__(config)
        self.bert = TFBertMainLayer(config, name="bert")

        # Your custom layers
        self.classifier = tf.keras.layers.Dense(
            units=1,  # For binary classification
            kernel_initializer=tf.keras.initializers.get(config.initializer_range),
            name="classifier"
        )
        self.log_variance_predictor = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.get(config.initializer_range),
            name="log_variance"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(
            self,
            input_ids: TFModelInputType | None = None,
            attention_mask: np.ndarray | tf.Tensor | None = None,
            token_type_ids: np.ndarray | tf.Tensor | None = None,
            position_ids: np.ndarray | tf.Tensor | None = None,
            head_mask: np.ndarray | tf.Tensor | None = None,
            inputs_embeds: np.ndarray | tf.Tensor | None = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,  # None
            labels: np.ndarray | tf.Tensor | None = None,
            training: Optional[bool] = False,
    ) -> CustomTFSequenceClassifierOutput:
        outputs = self.bert.call(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
            labels,
            training,
        )
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output, training=training)

        logits = self.classifier(pooled_output)
        log_variances = self.log_variance_predictor(pooled_output)

        loss = None
        if labels is not None:
            if self.custom_loss_fn is not None:
                loss = self.loss_fn(labels, logits, log_variances)
            else:
                loss = self.loss_fn(labels, logits)
        return CustomTFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            log_variances=log_variances,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    """ 
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.call(x, training=True)
            loss = self.compiled_loss(y, {'logits': y_pred.logits, 'log_variances': y_pred.log_variances})

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for metric in self.compiled_metrics:
            metric.update_state(y, y_pred.logits)

        return {m.name: m.result() for m in self.metrics}
    """


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
            y_pred = self.call(x, training=True)
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


def create_bert_config(hidden_dropout_prob, attention_probs_dropout_prob, classifier_dropout):
    config = BertConfig()
    config.hidden_dropout_prob = hidden_dropout_prob
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.classifier_dropout = classifier_dropout

    if not isinstance(hidden_dropout_prob, float) or not 0 <= hidden_dropout_prob < 1:
        raise ValueError("hidden_dropout_prob must be a float in the range [0, 1).")

    return config


class StudentBody(tf.keras.Model):
    def __init__(self, bert_base):
        super().__init__()
        self.bert_base = bert_base

    def call(self, inputs, training=False, mask=None):
        bert_outputs = self.bert(inputs, training=training)
        pooled_output = bert_outputs.pooler_output
        return pooled_output


class StudentHead(tf.keras.Model):
    def __init__(self, classifier_head, log_variance_head, dropout_rate=0.1):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier_head = classifier_head
        self.log_variance_predictor = log_variance_head

    def call(self, inputs, training=False, mask=None):
        dropout_output = self.dropout(inputs, training=training)

        logits = self.classifier_head(dropout_output)
        probs = tf.nn.sigmoid(logits)
        log_variances = self.log_variance_predictor(dropout_output)

        return CustomTFSequenceClassifierOutput(
            logits=logits,
            probs=probs,
            log_variances=log_variances,
        )


class MCDropoutBERTStudent(tf.keras.Model):
    def __init__(self, student_model, dropout_rate):
        super().__init__()
        self.student_body = StudentBody(student_model.bert)
        self.student_head = StudentHead(student_model.classifier, student_model.log_variance_predictor, dropout_rate)

    def cached_mc_dropout_predict(self, inputs, n=20) -> dict:
        student_body_outputs = self.student_body(inputs, training=False)  # cached output of BERT base

        all_logits = []
        all_probs = []
        all_log_variances = []
        for i in range(n):
            tf.random.set_seed(range(n)[i])
            outputs = self.student_head(student_body_outputs, training=True)
            logits = outputs['logits']
            probs = outputs['probs']
            log_variances = outputs['log_variances']
            all_logits.append(logits)
            all_probs.append(probs)
            all_log_variances.append(log_variances)

        all_logits = tf.stack(all_logits, axis=0)
        all_probs = tf.stack(all_probs, axis=0)
        all_log_variances = tf.stack(all_log_variances, axis=0)
        mean_predictions = tf.reduce_mean(all_logits, axis=0)
        var_predictions = tf.math.reduce_variance(all_logits, axis=0)

        # TODO: validate that this works as intended

        return {'logits': all_logits,
                'probs': all_probs,
                'log_variances': all_log_variances,
                'mean_predictions': mean_predictions,
                'var_predictions': var_predictions,
                }
