"""
Using the preprocessed data, we now convert the data into a format that can be used by the BERT model.
"""
from typing import Tuple

from transformers import BertTokenizer
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 48  # in accordance with max length of the data sequences


def bert_tokenize(all_sequences, max_length) -> Tuple[list, list]:
    input_ids = []
    attention_masks = []
    for sequence in all_sequences:
        bert_inp = tokenizer(text=sequence, max_length=max_length, padding='max_length', pad_to_max_length=True,
                             truncation=True, return_token_type_ids=False)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    return input_ids, attention_masks


def bert_preprocess(preprocessed_data, max_length=MAX_LEN) -> Tuple[list, list, list]:
    """
    Preprocesses the data for BERT.
    :param preprocessed_data: The preprocessed data.
    :param max_length: The maximum length of the sequences.
    :return: Tuple of input_ids, attention_masks, labels
    """
    sequences, labels = preprocessed_data['text'].values, preprocessed_data['target'].values
    input_ids, attention_masks = bert_tokenize(sequences, max_length)
    labels = list(labels)

    return input_ids, attention_masks, labels
