import random
from random import shuffle

# first time using wordnet:
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet


random.seed(1)


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
    TODO: replace get synonyms by random word

    For each word in the sentence, with a probability p, insert a randomly chosen word
    either before or after the word in the sentence.

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

def introduce_noise(sequence, p_sr=0, p_ri=0, p_rs=0, p_rd=0):
    words = sequence.split(' ')
    words = [word for word in words if word is not '']

    augmented_sequence = None

    if p_sr > 0:
        a_words = synonym_replacement(words, p_sr)
        augmented_sequence = ' '.join(a_words)

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
