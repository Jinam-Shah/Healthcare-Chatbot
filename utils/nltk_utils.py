import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Initialize tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

# Medical stopwords to keep (don't remove these)
MEDICAL_KEEP = {'pain', 'not', 'no', 'cannot', 'cant', 'feel', 'have', 'had'}


def clean_text(text):
    """
    Clean and normalize text

    Args:
        text: Input string
    Returns:
        cleaned: Cleaned string
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special characters but keep medical symbols
    text = re.sub(r'[^a-zA-Z0-9\s\.\?\!°\-]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def tokenize(sentence):
    """
    Split sentence into array of words/tokens

    Args:
        sentence: Input string
    Returns:
        tokens: List of tokens
    """
    # Clean text first
    sentence = clean_text(sentence)

    # Tokenize
    tokens = nltk.word_tokenize(sentence)

    return tokens


def stem(word):
    """
    Stemming: Find the root form of the word

    Examples:
        organize, organizes, organizing -> organ

    Args:
        word: Input word
    Returns:
        stemmed: Stemmed word
    """
    return stemmer.stem(word.lower())


def lemmatize(word):
    """
    Lemmatization: Find the dictionary form of the word

    Examples:
        running, ran -> run

    Args:
        word: Input word
    Returns:
        lemmatized: Lemmatized word
    """
    return lemmatizer.lemmatize(word.lower())


def remove_stopwords(words, keep_medical=True):
    """
    Remove common stopwords (but keep medical terms)

    Args:
        words: List of words
        keep_medical: Keep medical stopwords
    Returns:
        filtered: Filtered word list
    """
    if keep_medical:
        return [w for w in words if w.lower() not in STOPWORDS or w.lower() in MEDICAL_KEEP]
    else:
        return [w for w in words if w.lower() not in STOPWORDS]


def bag_of_words(tokenized_sentence, all_words):
    """
    Return bag of words array (LEGACY - for backward compatibility)

    Args:
        tokenized_sentence: List of words
        all_words: Vocabulary list
    Returns:
        bag: Binary array (1 if word exists, 0 otherwise)
    """
    # Stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]

    # Initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1

    return bag


def preprocess_sentence(sentence, use_stemming=True, remove_stops=False):
    """
    Complete preprocessing pipeline

    Args:
        sentence: Input string
        use_stemming: Apply stemming (vs lemmatization)
        remove_stops: Remove stopwords
    Returns:
        processed: List of processed words
    """
    # Tokenize
    words = tokenize(sentence)

    # Remove stopwords
    if remove_stops:
        words = remove_stopwords(words)

    # Stem or lemmatize
    if use_stemming:
        words = [stem(w) for w in words]
    else:
        words = [lemmatize(w) for w in words]

    return words


def get_word_variations(word):
    """
    Get common variations of a word (for data augmentation)

    Args:
        word: Input word
    Returns:
        variations: Set of word variations
    """
    variations = {word}

    # Add stemmed version
    variations.add(stem(word))

    # Add lemmatized version
    variations.add(lemmatize(word))

    return variations
