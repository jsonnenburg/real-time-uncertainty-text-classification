# remove stopwords
from nltk.corpus import stopwords

stopwords = stopwords.words("english")

# following Davidson et al. (2017)
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)


def remove_stopwords(self, text: str) -> str:
    return " ".join([word for word in text.split() if word not in self.stopwords])

# tokenization / stemming, ...