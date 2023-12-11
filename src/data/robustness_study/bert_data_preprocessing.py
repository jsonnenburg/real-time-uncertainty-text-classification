from typing import Tuple

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
