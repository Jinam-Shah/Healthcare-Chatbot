import torch
import torch.nn as nn
import numpy as np
from gensim.models import Word2Vec
import pickle
import os


class EmbeddingLayer(nn.Module):
    """
    Trainable word embedding layer with optional pre-trained initialization
    """

    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None):
        super(EmbeddingLayer, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Initialize with pre-trained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True  # Still trainable
        else:
            # Xavier initialization
            nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - word indices
        Returns:
            (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(x)


class Word2VecTrainer:
    """
    Train Word2Vec embeddings on medical corpus
    """

    def __init__(self, embedding_dim=128, window=5, min_count=1):
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.model = None
        self.word2idx = {}
        self.idx2word = {}

    def train(self, sentences, save_path=None):
        """
        Train Word2Vec on tokenized sentences

        Args:
            sentences: List of tokenized sentences [[word1, word2], [word3, word4]]
            save_path: Path to save trained embeddings
        """
        print(f"Training Word2Vec on {len(sentences)} sentences...")

        # Train Word2Vec
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            sg=1,  # Skip-gram
            epochs=20
        )

        # Create word-to-index mapping
        self.word2idx = {word: idx + 1 for idx, word in enumerate(self.model.wv.index_to_key)}
        self.word2idx['<PAD>'] = 0  # Padding token
        self.word2idx['<UNK>'] = len(self.word2idx)  # Unknown token

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        print(f"Vocabulary size: {len(self.word2idx)}")

        # Save if path provided
        if save_path:
            self.save(save_path)

        return self.get_embedding_matrix()

    def get_embedding_matrix(self):
        """
        Convert Word2Vec to numpy matrix for PyTorch

        Returns:
            (vocab_size, embedding_dim) numpy array
        """
        vocab_size = len(self.word2idx)
        embedding_matrix = np.zeros((vocab_size, self.embedding_dim))

        for word, idx in self.word2idx.items():
            if word in ['<PAD>', '<UNK>']:
                continue
            if word in self.model.wv:
                embedding_matrix[idx] = self.model.wv[word]
            else:
                # Random for unknown
                embedding_matrix[idx] = np.random.normal(0, 0.1, self.embedding_dim)

        return embedding_matrix

    def save(self, path):
        """Save embeddings and vocab"""
        data = {
            'embedding_matrix': self.get_embedding_matrix(),
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'embedding_dim': self.embedding_dim
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Embeddings saved to {path}")

    @staticmethod
    def load(path):
        """Load pre-trained embeddings"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"Embeddings loaded from {path}")
        return data
