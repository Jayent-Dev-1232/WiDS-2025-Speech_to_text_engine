# Importing necessary libraries
import torch
from torch.utils.data import Dataset
from collections import Counter
import numpy as np

# Vocabulary class to handle word to index mapping
class Vocabulary:
    # Initialize vocabulary with special tokens
    def __init__(self, max_size=20000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    # Build vocabulary from a list of texts
    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        words = [w for w, c in counter.items() if c >= self.min_freq]
        words = words[: self.max_size - 2]

        for idx, word in enumerate(words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    # Encode text to a list of indices
    def encode(self, text):
        return [
            self.word2idx.get(word, self.word2idx["<UNK>"])
            for word in text.split()
        ]

# Dataset class for text data
class TextDataset(Dataset):
    # Initialize dataset with texts, labels, vocabulary, and max length
    def __init__(self, texts, labels, vocab, max_len=300):
        self.vocab = vocab
        self.max_len = max_len
        self.texts = texts
        self.labels = labels

        self.encoded = [self.pad(vocab.encode(t)) for t in texts]

    # Pad sequences to the maximum length
    def pad(self, seq):
        if len(seq) >= self.max_len:
            return seq[:self.max_len]
        return seq + [0] * (self.max_len - len(seq))

    # Return the length of the dataset
    def __len__(self):
        return len(self.labels)

    # Get item by index
    def __getitem__(self, idx):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )