import re
import html
from typing import Optional, Tuple
from enum import Enum
# import nltk
# nltk.download('words')
from nltk.corpus import words

from bs4 import BeautifulSoup
import emoji

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# remove stopwords
from nltk.corpus import stopwords

stopwords_set = set(stopwords.words("english"))
other_exclusions = {"#ff", "ff", "rt"}
stopwords_set.update(other_exclusions)


class DataLoader:

    def __init__(self, data_dir: str):

        self.data_dir: str = data_dir
        self.data: Optional[pd.DataFrame] = None

    def load_data(self):
        self.data = pd.read_csv(self.data_dir, sep=',', index_col=0)
        self.data['target'] = ((self.data['class'] == 0) | (self.data['class'] == 1)).astype(int)
        self.data.drop(['hate_speech', 'offensive_language', 'neither', 'class', 'count'], axis=1, inplace=True)
        self.data.columns = ['text', 'target']

    def split(self, train_size: float, val_size: float, test_size: float) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert train_size + val_size + test_size == 1

        # first split: train and temp
        train, temp = train_test_split(self.data, train_size=train_size, stratify=self.data['target'],
                                       random_state=SEED)

        # adjust val_size proportion to account for the reduced dataset size
        val_size_adjusted = val_size / (val_size + test_size)

        # second split: val and test
        val, test = train_test_split(temp, train_size=val_size_adjusted, stratify=temp['target'], random_state=SEED)

        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class EntityTags(str, Enum):
    """
    Following mozafari2020, we encode the following entities:
    < user >, < number >, < hashtag >, < url >, < emoticon >.
    Additionally, we also encode emojis as < emoji >.
    """
    USER = '<user> ',
    NUMBER = '<number> ',
    HASHTAG = '<hashtag> ',
    URL = '<url> ',
    EMOTICON = '<emoticon> ',
    EMOJI = ' <emoji> '

    def __str__(self):
        return self.value


ENTITY_PLACEHOLDERS = {
    "<user>": "USERENTITY",
    "<number>": "NUMBERENTITY",
    "<hashtag>": "HASHTAGENTITY",
    "<url>": "URLENTITY",
    "<emoticon>": "EMOTICONENTITY",
    "<emoji>": "EMOJIENTITY"
}


# precompile all regular expressions
regex_user = re.compile(r'@\w+')
regex_number = re.compile(r'\d+')
regex_hashtag = re.compile(r'#\w+')
regex_url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%\d{2}[a-fA-F]))+')
regex_emoticon = re.compile(r'(:\)|:\(|:D|:P)')
regex_punctuation = re.compile(r'[^\w\s]')
regex_unknown_unicode = re.compile(r'\\u[0-9a-fA-F]{4}')
regex_multiple_spaces = re.compile("\s\s+")
regex_quotes = re.compile(r'["“”]+')
regex_newlines = re.compile(r'[\r\n]+')
regex_elongated = re.compile(r'(.)\1{2,}')


word_list = set(words.words())


def split_hashtag(hashtag,):
    s = hashtag
    result = []

    while s:
        found = False
        for i in range(len(s), 0, -1):
            if s[:i].lower() in word_list:
                result.append(s[:i])
                s = s[i:]
                found = True
                break
        if not found:
            result.append(s)
            break

    return ' '.join(result)


def replace_entities(text):
    text = regex_user.sub(EntityTags.USER, text)
    text = regex_url.sub(EntityTags.URL, text)
    text = regex_number.sub(EntityTags.NUMBER, text)
    text = regex_hashtag.sub(lambda m: EntityTags.HASHTAG + split_hashtag(m.group(0)[1:]), text)
    text = regex_emoticon.sub(lambda m: EntityTags.EMOTICON, text)
    text = emoji.replace_emoji(text, replace=str(EntityTags.EMOJI))

    return text


def replace_elongated_words(text):
    """
    Replacing three or more consecutive identical characters with one.
    """
    return regex_elongated.sub(r'\1', text)


def remove_multiple_spaces(text: str) -> str:
    return regex_multiple_spaces.sub(" ", text)


def remove_quotes(text) -> str:
    return regex_quotes.sub('', text)


def remove_newlines(text) -> str:
    """
    Replace occurrences of \r, \n, or \r\n (in any combination) with a single space.
    """
    return regex_newlines.sub(' ', text)


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation, but keep the entities intact.
    """
    # replace entities with placeholders
    for entity, placeholder in ENTITY_PLACEHOLDERS.items():
        text = text.replace(entity, placeholder)

    # remove punctuation
    text = regex_punctuation.sub('', text)

    # restore entities
    for entity, placeholder in ENTITY_PLACEHOLDERS.items():
        text = text.replace(placeholder, entity)

    return text


def remove_unknown_unicodes(text: str) -> str:
    return regex_unknown_unicode.sub('', text)


def clean_html_content(text: str) -> str:
    return BeautifulSoup(html.unescape(text), "html.parser").get_text()


def preprocess(text: str) -> str:
    """
    All steps following mozafari2020 (4.2).
    """
    text = remove_newlines(text)
    text = clean_html_content(text)
    text = remove_quotes(text)
    text = replace_elongated_words(text)
    text = replace_entities(text)
    text = remove_punctuation(text)
    text = remove_multiple_spaces(text)
    text = text.lower()
    return text


def remove_stopwords(text: str) -> str:
    return " ".join(word for word in text.split() if word not in stopwords_set)
