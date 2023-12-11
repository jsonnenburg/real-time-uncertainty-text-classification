from __future__ import annotations

from typing import Optional, Union, Tuple, Any

import numpy as np
from tensorflow import Tensor
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
    def __init__(self, labels=None, loss=None, logits=None, hidden_states=None, attentions=None, log_variances=None):
        super().__init__(labels=labels, loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
        self.labels = labels
        self.log_variances = log_variances


class MCDropoutBERTDoubleHead(MCDropoutTFBertForSequenceClassification):

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
    ) -> Tuple[Tensor | None, Tensor, Any] | CustomTFSequenceClassifierOutput:
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

        if len(outputs) > 1:
            pooled_output = outputs[1]
        else:
            print(type(outputs))
            print(len(outputs))
            raise ValueError("Expected pooled output not found in model outputs.")

        log_variances = self.log_variance_predictor(pooled_output)

        if not return_dict:
            loss = outputs.loss if labels is not None else None
            logits = outputs.logits
            return loss, logits, log_variances

        return CustomTFSequenceClassifierOutput(
            labels=labels,
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            log_variances=log_variances,
        )

def create_bert_config(hidden_dropout_prob, attention_probs_dropout_prob, classifier_dropout):
    config = BertConfig()
    config.hidden_dropout_prob = hidden_dropout_prob
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.classifier_dropout = classifier_dropout
    return config


#####################
# Example usage

from src.utils.loss_functions import aleatoric_loss, shen_loss

tokenized_training_data = ...
tokenized_val_data = ...

# the optimizer to be used for training
optimizer = ...
# the metrics to be recorded during training
metrics = ...

# grid-search over layer-wise dropout probabilities
hidden_dropout_probs = [0.1, 0.2, 0.3]
attention_dropout_probs = [0.1, 0.2, 0.3]
classifier_dropout_probs = [0.1, 0.2, 0.3]

batch_size = ...
epochs = ...

for hidden_dropout in hidden_dropout_probs:
    for attention_dropout in attention_dropout_probs:
        for classifier_dropout in classifier_dropout_probs:
            config = create_bert_config(hidden_dropout, attention_dropout, classifier_dropout)
            # Initialize and train your model with this config
            model = MCDropoutBERTDoubleHead.from_pretrained('bert-base-uncased', config=config)
            model.compile(optimizer=optimizer, loss=aleatoric_loss, metrics=metrics)
            model.fit(tokenized_training_data, epochs=epochs, validation_data=tokenized_val_data, batch_size=batch_size)
            # Evaluate the model and record the performance metrics
            # in terms of classification loss
            # eval on val set, record metrics, save model checkpoint and config
            model(tokenized_val_data, training=False)
            # ...

# load the best model checkpoint and config
config = ...
# this is the teacher model
bert_teacher = ...

# train on training + validation data

# evaluate on test set


# save the teacher model
bert_teacher.save_weights('path/to/save/model_weights.h5')
# save the optimizer state
optimizer.save_weights('path/to/save/optimizer_weights.h5')

# initialize the student model
bert_student = MCDropoutBERTDoubleHead.from_pretrained('bert-base-uncased', config=config)
bert_student.set_weights(bert_teacher.get_weights())


# obtain samples from the teacher model for the training data (with training=True) and SAVE THEM
# for each training sequence, generate m predictive samples (tuples of logits, log variance) using MC dropout from teacher
# over m samples, compute average observation noise for each sequence
# obtain final predictive samples for each sequence by sampling from a Gaussian with mean = logit and variance = average observation noise * std normal (afaik)
# ^do this k times for each of the m predictive samples to obtain m * k predictive samples for each training sequence
# hence, the training data is now m * k times larger than before --> training process is O(m * k) times slower
def generate_student_training_samples(teacher, training_data, m=5, k=10):
    """

    :param teacher:
    :param training_data:
    :param m:
    :param k:
    :return: student training data, i.e. tuples of (ground truth label, predictive sample)
    """
    student_training_samples = []
    for sequence, ground_truth_label in training_data:
        predictive_samples = []
        for _ in range(m):
            predictive_samples.append(teacher(sequence, training=True))  # teacher outputs logit, log variance!
            # compute average observation noise
            avg_observation_noise = ...
            for _ in range(k):
                # sample from Gaussian with mean = logit and variance = average observation noise * std normal
                mean = ...
                log_variance = ...
                eps = tf.random.normal(shape=[k])
                predictive_sample = mean + tf.sqrt(tf.exp(log_variance)) * eps
                sample = ground_truth_label, predictive_sample
                student_training_samples.append(sample)
    return student_training_samples


# train student on mean, log_variance pairs using the samples from the teacher model and shen loss
# ...

# obtain student predictions on test set (MC dropout) using training=False (which keeps dropout active for last layer)
# - want to cache rest of the model (i.e. the BERT part) and only recompute the last layer

# to implement "cache", split student model into two parts: BERT part and last layer
class StudentBody(tf.keras.Model):
    def __init__(self, bert_base):
        super().__init__()
        self.bert_base = bert_base  # BERT student's base layers

    def call(self, inputs):
        # Extract the last hidden states
        outputs = self.bert_base(inputs)[0]  # The shape of outputs is (batch_size, sequence_length, hidden_size)

        # Maybe you want to use the pooled output (representing [CLS] token)
        pooled_output = outputs[:, 0, :]  # shape: (batch_size, hidden_size)

        return pooled_output


class StudentHead(tf.keras.Model):

    def __init__(self, classifier_head, log_variance_head, dropout_rate=0.1):
        super().__init__()

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.classifier_head = classifier_head
        self.log_variance_predictor = log_variance_head

    def call(self, inputs, training=True):
        # Apply dropout for MC Dropout during both training and inference
        dropout_output = self.dropout(inputs, training=training)  # Force dropout

        # Apply classification heads
        logits = self.classifier_head(dropout_output)
        log_variance = self.classifier2(dropout_output)

        return logits, log_variance


class MCDropoutBERTStudent:
    """Student model, only used for inference.

    The student model consists of a BERT base model and a last layer with MC Dropout enabled.
    """
    def __init__(self, student, dropout_rate):
        self.student = student

        self.student_body = StudentBody(self.student.bert)
        self.student_head = StudentHead(self.student.classifier, self.student.log_variance_predictor, dropout_rate)

    def predict(self, inputs, n=20):
        student_body_outputs = self.student_body(inputs, training=False)
        # feed cached BERT outputs into last layer and obtain MC dropout samples
        student_predictions = [self.student_head(student_body_outputs, training=True) for _ in range(n)]
        return student_predictions


data_test_preprocessed = ...

mc_dropout_student = MCDropoutBERTStudent(bert_student, dropout_rate=0.1)
mc_dropout_student.predict(data_test_preprocessed, n=20)


# TODO: this approach means that we can get rid of MCDropoutTFBertForSequenceClassification and instead use the \
#  standard BERT model from which MCDropoutBERTDoubleHead inherits

