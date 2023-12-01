import re
import html
from typing import Optional, Tuple

from bs4 import BeautifulSoup
import emoji

import pandas as pd

SEED = 42


class DataLoader:

    def __init__(self, data_dir: str):

        self.data_dir: str = data_dir
        self.data: Optional[pd.DataFrame] = None

    def load_data(self):
        self.data = pd.read_csv(self.data_dir, sep=',', index_col=0)
        self.data['target'] = ((self.data['class'] == 0) | (self.data['class'] == 1)).astype(int)
        self.data.drop(['hate_speech', 'offensive_language', 'neither', 'class', 'count'], axis=1, inplace=True)
        self.data.columns = ['text', 'target']

    def split(self, train_size: float, val_size: float, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert train_size + val_size + test_size == 1
        self.data = self.data.sample(frac=1, random_state=SEED).reset_index(drop=True)
        train_end = int(train_size * len(self.data))
        val_end = int((train_size + val_size) * len(self.data))
        train = self.data.iloc[:train_end].reset_index(drop=True)
        val = self.data.iloc[train_end:val_end].reset_index(drop=True)
        test = self.data.iloc[val_end:].reset_index(drop=True)
        return train, val, test


def replace_entities(text):
    text = re.sub(r'@\w+', 'user ', text)

    text = re.sub(r'\d+', 'number ', text)

    text = re.sub(r'#\w+', lambda m: 'hashtag ' + m.group(0)[1:], text)

    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%\d{2}[a-fA-F]))+', 'url ', text)

    text = re.sub(r'(:\)|:\(|:D|:P)', lambda m: 'emoticon ', text)

    text = emoji.replace_emoji(text, replace='emoji ')

    return text


def replace_elongated_words(text):
    """
     Replacing three or more consecutive identical characters with one.
    """
    return re.sub(r'(.)\1{2,}', r'\1', text)


def remove_multiple_spaces(text: str) -> str:
    return re.sub("\s\s+", " ", text)


def remove_quotes(text) -> str:
    return re.sub(r"[\"“”']", '', text)


def remove_newlines(text) -> str:
    """Replace occurrences of \r, \n, or \r\n (in any combination) with a single space.
    """
    return re.sub(r'[\r\n]+', ' ', text)


def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)


def clean_html_content(text: str) -> str:
    return BeautifulSoup(html.unescape(text), "html.parser").get_text()


def preprocess(text: str) -> str:
    text = remove_newlines(text)
    text = clean_html_content(text)
    text = remove_quotes(text)
    text = replace_elongated_words(text)
    text = replace_entities(text)
    text = remove_punctuation(text)
    text = remove_multiple_spaces(text)
    text = text.lower()
    return text
