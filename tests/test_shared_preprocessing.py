import pytest

from src.data.shared_data_preprocessing import GeneralPreprocessor

@pytest.fixture
def preprocessor():
    return GeneralPreprocessor()


def test_remove_stopwords(preprocessor):
    text = "this is a test"
    assert preprocessor.remove_stopwords(text) == "test"


def test_remove_multiple_spaces(preprocessor):
    text = "this  is a    test"
    assert preprocessor.remove_multiple_spaces(text) == "this is a test"


def test_replace_emoji(preprocessor):
    text = "this is a test 😂"
    assert preprocessor.replace_emoji(text) == "this is a test :face_with_tears_of_joy:"


def test_remove_new_line(preprocessor):
    text = "this is a test \n"
    assert preprocessor.remove_new_line(text) == "this is a test  "


def test_replace_hashtags(preprocessor):
    text = "this is a #test"
    assert preprocessor.replace_hashtags(text) == "this is a test"


def test_replace_mentions(preprocessor):
    text = "this is a @test"
    assert preprocessor.replace_mentions(text) == "this is a test"


def test_preprocess(preprocessor):
    text = "this is a @test \n #test 😂"
    assert preprocessor.preprocess(text) == "test test :face_with_tears_of_joy:"