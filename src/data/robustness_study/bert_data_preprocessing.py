from typing import Tuple, Dict

from transformers import BertTokenizer
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def bert_preprocess(preprocessed_data, max_length=48) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Preprocesses the data for BERT.
    :param preprocessed_data: The preprocessed data.
    :param max_length: The maximum length of the sequences.
    :return: Tuple of input_ids, attention_masks, labels
    """
    sequences, labels = preprocessed_data['text'].values, preprocessed_data['target'].values
    tokenized_output = tokenizer(
        text=sequences.tolist(),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

    input_ids = tokenized_output['input_ids']
    attention_masks = tokenized_output['attention_mask']
    labels = tf.constant(labels)

    return input_ids, attention_masks, labels


def get_tf_dataset(tokenized_dataset: Dict, subset: str) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': tokenized_dataset[subset][0],
            'attention_mask': tokenized_dataset[subset][1]
        },
        tokenized_dataset[subset][2]  # labels
    ))
    return dataset


def transfer_data_bert_preprocess(preprocessed_data, max_length=48) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Preprocesses the data for BERT student model transfer learning.
    :param preprocessed_data: The preprocessed data.
    :param max_length: The maximum length of the sequences.
    :return: Tuple of input_ids, attention_masks, labels
    """
    sequences, labels, predictions = preprocessed_data['sequences'].values, preprocessed_data['labels'].values, preprocessed_data['predictions'].values
    tokenized_output = tokenizer(
        text=sequences.tolist(),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

    input_ids = tokenized_output['input_ids']
    attention_masks = tokenized_output['attention_mask']
    labels = tf.constant(labels)
    predictions = tf.constant(predictions)

    return input_ids, attention_masks, labels, predictions


def transfer_get_tf_dataset(tokenized_dataset: Dict, subset: str) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': tokenized_dataset[subset][0],
            'attention_mask': tokenized_dataset[subset][1]
        },
        [tokenized_dataset[subset][2], tokenized_dataset[subset][3]]
    ))
    return dataset
