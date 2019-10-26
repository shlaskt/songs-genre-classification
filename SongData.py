from torch.utils.data import Dataset
import torch


class SongData(Dataset):
    def __init__(self, lyrics, labels, w2v):
        self.texts = lyrics
        self.labels = labels
        self.w2v = w2v

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = [self.w2v.wv.vocab[word].index for word in text if word in self.w2v.wv.vocab]
        label = self.labels[idx]
        return torch.tensor(text, dtype=torch.long), torch.tensor(label, dtype=torch.long)
