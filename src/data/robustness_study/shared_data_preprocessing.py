import re
import html
from typing import Optional, Tuple

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
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
        train = self.data.iloc[:train_end]
        val = self.data.iloc[train_end:val_end]
        test = self.data.iloc[val_end:]
        return train, val, test


stopwords = stopwords.words("english")

# following Davidson et al. (2017)
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)


class GeneralTextPreprocessor:

    def __init__(self):
        self.stopwords = stopwords
        self.other_exclusions = other_exclusions

    def remove_stopwords(self, text: str) -> str:
        return " ".join([word for word in text.split() if word not in self.stopwords])

    @staticmethod
    def remove_multiple_spaces(text: str) -> str:
        return re.sub("\s\s+", " ", text)

    @staticmethod
    def replace_emoji(text) -> str:
        return emoji.demojize(text, delimiters=("", ""))

    @staticmethod
    def remove_quotes(text) -> str:
        return re.sub(r"[\"“”']", '', text)

    @staticmethod
    def remove_newlines(text) -> str:
        """Replace occurrences of \r, \n, or \r\n (in any combination) with a single space.
        """
        return re.sub(r'[\r\n]+', ' ', text)

    @staticmethod
    def replace_hashtags(text: str) -> str:
        """Adapted from https://stackoverflow.com/questions/38506598/regular-expression-to-match-hashtag-but-not-hashtag-with-semicolon
        """
        return re.sub(r'\B(\#[a-zA-Z]+\b)(?!;)', 'HASHTAGHERE', text)

    @staticmethod
    def replace_mentions(text: str) -> str:
        return re.sub(r'@\w+', 'MENTIONHERE', text)

    @staticmethod
    def replace_urls(text: str) -> str:
        """Adapted from Davidson et al. (2017).
        """
        url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%\d{2}[a-fA-F]))+'
        return re.sub(url_regex, 'URLHERE', text)

    @staticmethod
    def clean_html_content(text: str) -> str:
        decoded_text = html.unescape(text)
        return BeautifulSoup(decoded_text, "html.parser").get_text()

    def preprocess(self, text: str) -> str:
        text = self.remove_stopwords(text)
        text = self.remove_newlines(text)
        text = self.remove_multiple_spaces(text)
        text = self.replace_hashtags(text)
        text = self.replace_mentions(text)
        text = self.replace_urls(text)
        text = self.clean_html_content(text)
        text = self.replace_emoji(text)
        text = self.remove_quotes(text)
        text = text.lower()
        return text
