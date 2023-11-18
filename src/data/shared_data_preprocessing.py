import re
import string
from nltk.corpus import stopwords
import emoji


stopwords = stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)


class GeneralPreprocessor:

    def __init__(self):
        self.stopwords = stopwords
        self.other_exclusions = other_exclusions

    def remove_stopwords(self, text: str):
        return " ".join([word for word in text.split() if word not in self.stopwords])

    @staticmethod
    def remove_multiple_spaces(text: str):
        return re.sub("\s\s+" , " ", text)

    @staticmethod
    def replace_emoji(text):
        return emoji.replace_emoji(text)

    @staticmethod
    def remove_new_line(text: str):
        return text.replace('\r\n', ' ').replace('\n', ' ')

    @staticmethod
    def replace_hashtags(text: str):
        # TODO: improve this
        return text.replace('#', '')

    @staticmethod
    def replace_mentions(text: str):
        # TODO: improve this
        return text.replace('@', '')

    def preprocess(self, text: str):
        text = self.remove_stopwords(text)
        text = self.remove_new_line(text)
        text = self.remove_multiple_spaces(text)
        text = self.replace_emoji(text)
        text = self.replace_hashtags(text)
        text = self.replace_mentions(text)
        text = text.lower()
        return text


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text


#Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower() #remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text

def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2
