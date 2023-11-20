from __future__ import annotations

from typing import Optional, Union, Tuple

import numpy as np
from transformers import TFBertForSequenceClassification, BertConfig
import tensorflow as tf
from transformers.modeling_tf_outputs import TFSequenceClassifierOutput
from transformers.modeling_tf_utils import TFModelInputType


class MCDropoutTFBertForSequenceClassification(TFBertForSequenceClassification):
    """A standard BERT model with MC Dropout added to the last layer (always active).
    """
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
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(inputs=pooled_output, training=True)
        logits = self.classifier(inputs=pooled_output)
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class CustomTFSequenceClassifierOutput(TFSequenceClassifierOutput):
    def __init__(self, loss=None, logits=None, hidden_states=None, attentions=None, log_variance=None):
        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
        self.log_variance = log_variance


class MCDropoutStudent(MCDropoutTFBertForSequenceClassification):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.log_variance_predictor = tf.keras.layers.Dense(1)

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
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
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
        log_variance = self.log_variance_predictor(pooled_output)

        if not return_dict:
            return outputs + (log_variance,)

        return CustomTFSequenceClassifierOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            log_variance=log_variance,
        )



#####################
# Example usage

# Load pre-trained model
config = BertConfig.from_pretrained('bert-base-uncased')

config.hidden_dropout_prob = 0.2
config.attention_probs_dropout_prob = 0.2
config.classifier_dropout = 0.2

bert_teacher = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)


# Add MC Dropout
# Assuming dropout layers are added in the BERT model, ensure they're active during inference
# This might involve customizing the model class to override the call method
def add_mc_dropout(model, last_layer_only=True):
    # Function to modify the BERT model to include MC dropout
    # Differentiate between last layer only and all layers
    # ...
    # How to tune layer-wise dropout probabilities?
    # via config!
    # ...
    return model


# Clone the teacher model architecture
bert_student = MCDropoutStudent.from_pretrained('bert-base-uncased', config=config)
# initialize with teacher weights from saved checkpoint
bert_student.set_weights(bert_teacher.get_weights())

# train on mean, log_variance pairs
# ...

