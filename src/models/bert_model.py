from __future__ import annotations

from typing import Optional, Tuple, Any

import numpy as np
from tensorflow import Tensor
from transformers import TFBertForSequenceClassification, TFBertMainLayer, BertConfig
import tensorflow as tf
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput
from transformers.modeling_tf_utils import get_initializer, TFModelInputType


class CustomTFSequenceClassifierOutput(TFSequenceClassifierOutput):
    def __init__(self, labels=None, loss=None, logits=None, hidden_states=None, attentions=None, log_variances=None):
        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
        self.labels = labels
        self.log_variances = log_variances


class MCDropoutBERT(TFBertForSequenceClassification):
    def __init__(self, config, custom_loss_fn=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.bert = TFBertMainLayer(config, name="bert")
        # replace classifier with two separate heads
        self.classifier = tf.keras.layers.Dense(
            units=2, # binary classifier
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
            self.custom_loss_fn = tf.keras.losses.tf.nn.sparse_softmax_cross_entropy_with_logits()

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
            return_dict: Optional[bool] = None,
            labels: np.ndarray | tf.Tensor | None = None,
            training: Optional[bool] = False,
    ) -> Tuple[Tuple[Any, ...] | Any, Tensor] | CustomTFSequenceClassifierOutput:
        outputs = super().call(
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output, training=True)

        logits = self.classifier(pooled_output)
        log_variances: Tensor = self.log_variance_predictor(pooled_output)

        loss = None
        if labels is not None:
            loss = self.custom_loss_fn(labels, logits)

        if not return_dict:
            return (loss, logits) + outputs[2:], log_variances

        return CustomTFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            log_variances=log_variances,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


def create_bert_config(hidden_dropout_prob, attention_probs_dropout_prob, classifier_dropout):
    config = BertConfig()
    config.hidden_dropout_prob = hidden_dropout_prob
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.classifier_dropout = classifier_dropout
    return config
