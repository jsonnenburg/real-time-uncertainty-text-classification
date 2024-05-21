import pytest

from src.preprocessing.robustness_study.shared_data_preprocessing import *


def test_remove_multiple_spaces():
    text = "this  is a    test"
    assert remove_multiple_spaces(text) == "this is a test"


def test_replace_emoji():
    text = "this is a test 😂"
    assert replace_emoji(text) == "this is a test face_with_tears_of_joy"


def test_remove_newlines(preprocessor: GeneralTextPreprocessor):
    text = "this \r is a \r\r\r\n\n test \n"
    assert preprocessor.remove_newlines(text) == "this   is a   test  "


def test_replace_hashtags(preprocessor: GeneralTextPreprocessor):
    text = "this is a #test"
    assert preprocessor.replace_hashtags(text) == "this is a HASHTAGHERE"


def test_replace_mentions(preprocessor: GeneralTextPreprocessor):
    text = "this is a @test"
    assert preprocessor.replace_mentions(text) == "this is a MENTIONHERE"


def test_clean_html_content(preprocessor: GeneralTextPreprocessor):
    text = "this is a <html>test &#127867;&#127867;&#128540; </html>"
    assert preprocessor.clean_html_content(text) == "this is a test 🍻🍻😜 "


def test_remove_quotes(preprocessor: GeneralTextPreprocessor):
    text = "this is a \"test\""
    assert preprocessor.remove_quotes(text) == "this is a test"


def test_preprocess(preprocessor: GeneralTextPreprocessor):
    text = "this is a @test \n #test 😂 &#127867;"
    assert preprocessor.preprocess(text) == "mentionhere hashtaghere face_with_tears_of_joy clinking_beer_mugs"
