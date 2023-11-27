import pytest

from src.experiments.robustness_study.noise import *

random.seed(42)


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
