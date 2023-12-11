import pandas as pd
import pytest

from src.experiments.robustness_study.noise import *

random.seed(42)

data_test = pd.read_csv("/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/data/robustness_study/preprocessed/test.csv", sep="\t", index_col=0)


def test_pos_guided_word_replacement():
    word_distribution = WordDistributionByPOSTag(data_test['text'])
    sequence = 'this is a totally normal test sentence <hashtag>'
    p_pos = 1
    assert pos_guided_word_replacement(word_distribution, sequence, p_pos) == 'this give a already i bitch minute <hashtag>'

    sequence_tags_only = ' '.join(ENTITY_PLACEHOLDERS.keys())
    assert pos_guided_word_replacement(word_distribution, sequence_tags_only, p_pos) == sequence_tags_only


def test_synonym_replacement():
    words = ['this', 'is', 'a', '<hashtag>', 'test']
    p = 1
    new_words = synonym_replacement(words, p)
    assert new_words != words and '<hashtag>' in new_words

    tags_only = ENTITY_PLACEHOLDERS.keys()
    new_words = synonym_replacement(tags_only, p)
    assert new_words == [key for key in ENTITY_PLACEHOLDERS.keys()]


def test_random_insertion():
    words = ['this', 'is', 'a', 'test']
    p = 0.5
    new_words = random_insertion(words, p)
    assert new_words != words


def test_random_swap():
    words = ['this', 'is', 'a', 'test']
    p = 0.5
    new_words = random_swap(words, p)
    assert new_words == ['is', 'test', 'a', 'this']


# test random deletion
def test_single_word():
    assert random_deletion(["word"], 0.5) == ["word"]


def test_empty_list():
    assert random_deletion([], 0.5) == []


def test_high_deletion_probability():
    words = ["the", "quick", "brown", "fox"]
    deleted_words = random_deletion(words, 0.9)
    assert len(words) >= len(deleted_words) > 0


def test_low_deletion_probability():
    words = ["the", "quick", "brown", "fox"]
    deleted_words = random_deletion(words, 0.1)
    assert len(deleted_words) >= len(words) - 1


def test_zero_deletion_probability():
    words = ["the", "quick", "brown", "fox"]
    assert random_deletion(words, 0) == words


def test_full_deletion_probability():
    words = ["the", "quick", "brown", "fox"]
    deleted_words = random_deletion(words, 1)
    assert len(deleted_words) == 1
