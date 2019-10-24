import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Lstm(nn.Module):
    def __init__(self, word2vec):
        super(Lstm, self).__init__()
        self.word2vec = word2vec
        self.lstm = nn.Linear(300, 100)
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        sentence = torch.from_numpy(np.asarray([self.word2vec[word] for word in x if word in self.word2vec.wv.vocab]))
        # pred = sentence.reshape((1, sentence.shape[0], 300))
        pred = self.lstm(sentence)
        pred = self.fc(pred)
        return F.softmax(pred)

