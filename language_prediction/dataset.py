import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path

class ShakespearDataset(Dataset):

    def __init__(self, path, split, seq_len):
        super().__init__()

        self.path = path
        self.split = split
        self.seq_len = seq_len

        with open(self.path, 'r', encoding='utf-8') as f:
            text = f.read()

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
        self.decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

        # Train and test splits
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        s = int(self.split[0] * len(self.data))
        e = s + int((self.split[1] - self.split[0]) * len(self.data))
        self.data = self.data[s:e]

    def __len__(self):
        return len(self.data) - (self.seq_len + 1)

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_len]
        y = self.data[index+1:index+self.seq_len+1]
        return x, y

    def get_vocab_size(self):
        return self.vocab_size