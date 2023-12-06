from nltk.corpus import wordnet
import random
import nltk
# first time using wordnet or averaged_perceptron_tagger, download them
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

from src.data.robustness_study.shared_data_preprocessing import ENTITY_PLACEHOLDERS

from nltk.corpus import stopwords

stopwords = stopwords.words("english")

random.seed(1)

########################################################################
# POS-guided word replacement (tang2019)
# With probability p, we replace a word with another of the same POS tag. To preserve the original distribution,
# the new word is sampled from the unigram word distribution re-normalized by the part-of-speech (POS) tag. This rule
# perturbs the semantics of each example, e.g., “What do pigs eat?” is different from “How do pigs eat?”

# compute word distribution on test set! since we only perturb this subset
########################################################################


class WordDistributionByPOSTag:
    def __init__(self, test_set):
        self.test_set = test_set
        self.word_freq_by_pos = self.get_word_freq_by_pos(self.test_set)

    @staticmethod
    def nltk_to_wordnet_pos(nltk_pos):
        if nltk_pos.startswith('J'):
            return wordnet.ADJ
        elif nltk_pos.startswith('V'):
            return wordnet.VERB
        elif nltk_pos.startswith('N'):
            return wordnet.NOUN
        elif nltk_pos.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def get_word_freq_by_pos(self, data):
        word_freq_by_pos = {}
        for sentence in data:
            sentence = preprocess_sequence(sentence)
            tokenized_and_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
            for word, pos in tokenized_and_tagged:
                if word in ENTITY_PLACEHOLDERS.values():
                    continue
                wordnet_pos = self.nltk_to_wordnet_pos(pos)
                if wordnet_pos:
                    if wordnet_pos not in word_freq_by_pos:
                        word_freq_by_pos[wordnet_pos] = {}
                    if word in word_freq_by_pos[wordnet_pos]:
                        word_freq_by_pos[wordnet_pos][word] += 1
                    else:
                        word_freq_by_pos[wordnet_pos][word] = 1
        return word_freq_by_pos

    def get_words_with_same_pos(self, wordnet_pos):
        if wordnet_pos not in self.word_freq_by_pos:
            return []
        words_with_freq = self.word_freq_by_pos[wordnet_pos]
        words, frequencies = list(words_with_freq.keys()), list(words_with_freq.values())
        chosen_word = random.choices(words, weights=frequencies, k=1)[0]
        return chosen_word


def preprocess_sequence(sequence):
    for tag, placeholder in ENTITY_PLACEHOLDERS.items():
        sequence = sequence.replace(tag, placeholder)
    return sequence


def postprocess_sequence(sequence):
    for tag, placeholder in ENTITY_PLACEHOLDERS.items():
        sequence = sequence.replace(placeholder, tag)
    return sequence


def tokenize_and_pos_tag(sequence):
    sequence = preprocess_sequence(sequence)
    tagged_sequence = nltk.pos_tag(nltk.word_tokenize(sequence))
    return [(postprocess_sequence(word), pos) for word, pos in tagged_sequence]


def pos_guided_word_replacement(word_distribution, sequence, p_pos):
    """
    Replace each word in the sequence with another word of the same POS tag with probability p_pos.
    :param word_distribution:
    :param sequence:
    :param p_pos:
    :return: The sequence with some words replaced by words of the same POS tag.
    """
    tokenized_and_tagged = tokenize_and_pos_tag(sequence)
    new_sequence = []
    for word, pos in tokenized_and_tagged:
        if word in ENTITY_PLACEHOLDERS.keys() or word in stopwords:
            new_sequence.append(word)
            continue
        wordnet_pos = word_distribution.nltk_to_wordnet_pos(pos)
        word_replacement = word
        if random.random() < p_pos and wordnet_pos:
            same_pos_word = word_distribution.get_words_with_same_pos(wordnet_pos)
            word_replacement = same_pos_word if same_pos_word else word
        new_sequence.append(word_replacement)
    return ' '.join(new_sequence)


########################################################################

# Adapted from https://github.com/jasonwei20/eda_nlp/ (wei2019)

########################################################################
# Synonym replacement
########################################################################

def synonym_replacement(words, p):
    """
    Replace each word in the list with one of its synonyms with probability p.

    :param words: List of words to be potentially replaced.
    :param p: Probability of each word being replaced with a synonym.
    :return: List of words with some replaced by synonyms.
    """
    new_words = []

    for word in words:
        if word in ENTITY_PLACEHOLDERS.keys() or word in stopwords:
            new_words.append(word)
            continue
        if random.uniform(0, 1) <= p:
            synonyms = get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words.append(synonym)
            else:
                new_words.append(word)
        else:
            new_words.append(word)

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    if len(words) <= 1:
        return words

    new_words = [word for word in words if random.uniform(0, 1) > p]

    if not new_words:
        return [random.choice(words)]

    return new_words


########################################################################
# Random swap
########################################################################

def random_swap(words, p):
    """
    Swap p share of words in the list randomly.
    Probabilistic approach: For each word, with probability p, swap it with another word in the list.
    Will average out to p swaps per word.

    :param words: List of words to be swapped.
    :param p: Share of words that will be swapped.
    :return: List of words with swaps performed.
    """
    new_words = words.copy()

    for _ in range(len(words)):
        if random.uniform(0, 1) <= p:
            new_words = swap_word(new_words)

    return new_words


def swap_word(new_words):
    """
    Swap two words in a list at random indices.

    :param new_words: List of words where two words will be swapped.
    :return: List of words with two words swapped.
    """
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1

    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        # if we can't find a different random index after 3 tries, return the list as is
        if counter > 3:
            return new_words

    # perform the swap
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


########################################################################
# Random insertion
# Randomly insert words into the sentence with probability p
########################################################################

def random_insertion(words, p):
    """
    For each word in the sentence, with a probability p, find a random synonym and insert it either before or after the
    word in the sentence.

    :param words: List of words to insert into.
    :param p: Probability with which each word will have another word inserted nearby.
    :return: List of words with random insertions performed.
    """
    new_words = words.copy()

    i = 0
    while i < len(new_words):
        if random.uniform(0, 1) <= p:
            # randomly choose a word from the list to insert
            some_word = random.choice(new_words)
            synonyms = []
            counter = 0
            while len(synonyms) < 1:
                synonyms = get_synonyms(some_word)
                counter += 1
                if counter >= 10:
                    return new_words
            random_replacement = synonyms[0]
            # randomly choose whether to insert the word before or after the current position
            if random.uniform(0, 1) < 0.5:
                new_words.insert(i, random_replacement)
                i += 2  # move past the newly inserted word and the next original word
            else:
                new_words.insert(i + 1, random_replacement)
                i += 1  # move past the current word to the newly inserted word
        else:
            i += 1  # move past the current word, no insertion here

    return new_words


########################################################################
# main data augmentation function
########################################################################

def introduce_noise(sequence, word_distribution, p_sr=0, p_pr=0, p_ri=0, p_rs=0, p_rd=0):
    words = sequence.split(' ')
    words = [word for word in words if word != '']

    augmented_sequence = None

    if p_sr > 0:
        a_words = synonym_replacement(words, p_sr)
        augmented_sequence = ' '.join(a_words)

    if p_pr > 0:
        augmented_sequence = pos_guided_word_replacement(word_distribution, sequence, p_pr)

    if p_ri > 0:
        a_words = random_insertion(words, p_ri)
        augmented_sequence = ' '.join(a_words)

    if p_rs > 0:
        a_words = random_swap(words, p_rs)
        augmented_sequence = ' '.join(a_words)

    if p_rd > 0:
        a_words = random_deletion(words, p_rd)
        augmented_sequence = ' '.join(a_words)

    if augmented_sequence is not None:
        return augmented_sequence
    else:
        return sequence
