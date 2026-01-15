import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pickle
from collections import Counter
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nltk_utils import tokenize, stem


class DataProcessor:
    """
    Advanced data processing pipeline for chatbot training
    """

    def __init__(self, min_word_freq=1, max_seq_length=50, use_stemming=True):
        self.min_word_freq = min_word_freq
        self.max_seq_length = max_seq_length
        self.use_stemming = use_stemming

        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.tag2idx = {}
        self.idx2tag = {}

        self.all_words = []
        self.tags = []
        self.patterns = []

    def load_intents(self, intents_file):
        """
        Load and parse intents.json

        Args:
            intents_file: Path to intents.json
        Returns:
            intents: Parsed JSON data
        """
        print(f"Loading intents from {intents_file}...")
        with open(intents_file, 'r', encoding='utf-8') as f:
            intents = json.load(f)

        print(f"Loaded {len(intents['intents'])} intent categories")
        return intents

    def build_vocabulary(self, intents):
        """
        Build vocabulary from intents

        Args:
            intents: Parsed intents data
        Returns:
            xy: List of (tokenized_pattern, tag) tuples
        """
        print("\nBuilding vocabulary...")

        xy = []
        all_words_list = []

        # Extract patterns and tags
        for intent in tqdm(intents['intents'], desc="Processing intents"):
            tag = intent['tag']
            if tag not in self.tags:
                self.tags.append(tag)

            for pattern in intent['patterns']:
                # Tokenize
                words = tokenize(pattern)
                all_words_list.extend(words)
                xy.append((words, tag))

        # Stem and filter words
        if self.use_stemming:
            all_words_list = [stem(w) for w in all_words_list]

        # Count word frequencies
        word_freq = Counter(all_words_list)

        # Filter by minimum frequency
        self.all_words = sorted([w for w, freq in word_freq.items()
                                 if freq >= self.min_word_freq])

        # Build word2idx and idx2word
        for idx, word in enumerate(self.all_words, start=2):  # Start at 2 (0=PAD, 1=UNK)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        # Build tag2idx and idx2tag
        self.tags = sorted(set(self.tags))
        for idx, tag in enumerate(self.tags):
            self.tag2idx[tag] = idx
            self.idx2tag[idx] = tag

        print(f"\nVocabulary size: {len(self.word2idx)}")
        print(f"Number of tags: {len(self.tags)}")
        print(f"Number of patterns: {len(xy)}")

        return xy

    def encode_pattern(self, words):
        """
        Convert tokenized words to indices

        Args:
            words: List of words
        Returns:
            indices: List of word indices
        """
        if self.use_stemming:
            words = [stem(w) for w in words]

        indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        return indices

    def pad_sequence(self, sequence, max_length=None):
        """
        Pad sequence to max_length

        Args:
            sequence: List of indices
            max_length: Maximum length (default: self.max_seq_length)
        Returns:
            padded: Padded sequence
            length: Original length
        """
        if max_length is None:
            max_length = self.max_seq_length

        length = len(sequence)

        if length > max_length:
            # Truncate
            padded = sequence[:max_length]
            length = max_length
        else:
            # Pad with zeros
            padded = sequence + [0] * (max_length - length)

        return padded, length

    def prepare_training_data(self, xy):
        """
        Convert patterns to training tensors

        Args:
            xy: List of (words, tag) tuples
        Returns:
            X: Input sequences (batch, seq_len)
            y: Target labels (batch,)
            lengths: Sequence lengths (batch,)
        """
        print("\nPreparing training data...")

        X = []
        y = []
        lengths = []

        for words, tag in tqdm(xy, desc="Encoding sequences"):
            # Encode words to indices
            indices = self.encode_pattern(words)

            # Pad sequence
            padded, length = self.pad_sequence(indices)

            # Get tag index
            tag_idx = self.tag2idx[tag]

            X.append(padded)
            y.append(tag_idx)
            lengths.append(length)

        # Convert to numpy arrays
        X = np.array(X, dtype=np.int64)
        y = np.array(y, dtype=np.int64)
        lengths = np.array(lengths, dtype=np.int64)

        print(f"Training data shape: X={X.shape}, y={y.shape}")

        return X, y, lengths

    def get_tokenized_sentences(self, intents):
        """
        Get all tokenized sentences for Word2Vec training

        Args:
            intents: Parsed intents data
        Returns:
            sentences: List of tokenized sentences
        """
        sentences = []
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                words = tokenize(pattern)
                if self.use_stemming:
                    words = [stem(w) for w in words]
                sentences.append(words)
        return sentences

    def save(self, filepath):
        """Save processor state"""
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'tag2idx': self.tag2idx,
            'idx2tag': self.idx2tag,
            'all_words': self.all_words,
            'tags': self.tags,
            'max_seq_length': self.max_seq_length,
            'use_stemming': self.use_stemming
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data processor saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load processor state"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        processor = DataProcessor()
        processor.word2idx = data['word2idx']
        processor.idx2word = data['idx2word']
        processor.tag2idx = data['tag2idx']
        processor.idx2tag = data['idx2tag']
        processor.all_words = data['all_words']
        processor.tags = data['tags']
        processor.max_seq_length = data['max_seq_length']
        processor.use_stemming = data['use_stemming']

        print(f"Data processor loaded from {filepath}")
        return processor


class ChatDataset(Dataset):
    """
    PyTorch Dataset for chatbot training
    """

    def __init__(self, X, y, lengths):
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()
        self.lengths = torch.from_numpy(lengths).long()
        self.n_samples = len(X)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.lengths[index]

    def __len__(self):
        return self.n_samples
