"""
Using the preprocessed data, we now convert the data into a format that can be used by the BERT model.
"""
from typing import Tuple

from transformers import BertTokenizer
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 48  # in accordance with max length of the data sequences


def bert_tokenize(all_sequences, max_length) -> Tuple[np.ndarray, np.ndarray]:
    input_ids = []
    attention_masks = []
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for sequence in all_sequences:
        bert_inp = bert_tokenizer.__call__(sequence, max_length=max_length,
                                           padding='max_length', pad_to_max_length=True,
                                           truncation=True, return_token_type_ids=False)

        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])
    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)

    return input_ids, attention_masks


def bert_preprocess(preprocessed_data, max_length=MAX_LEN):
    sequences, labels = preprocessed_data['text'].values, preprocessed_data['target'].values
    input_ids, attention_masks = bert_tokenize(sequences, max_length)

    return input_ids, attention_masks, labels
