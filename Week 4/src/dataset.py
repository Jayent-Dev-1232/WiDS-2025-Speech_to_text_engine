import os
import torch
from torch.utils.data import Dataset
from features import extract_mfcc

class SpeechDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        for label, word in enumerate(["no", "yes"]):
            folder = os.path.join(root, word)
            for f in os.listdir(folder):
                self.samples.append((os.path.join(folder, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mfcc = extract_mfcc(path)
        return torch.tensor(mfcc).float(), torch.tensor(label)