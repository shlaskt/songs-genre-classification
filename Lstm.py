import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class Lstm(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights,
                 freeze_embeddings=False):
        super(Lstm, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(weights), freeze=freeze_embeddings)
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):

        input = self.word_embeddings(
            input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)

        if batch_size is None:
            h_0 = torch.zeros(1, self.batch_size, self.hidden_size)  # Initial hidden state of the LSTM
            c_0 = torch.zeros(1, self.batch_size, self.hidden_size)  # Initial cell state of the LSTM
        else:
            h_0 = torch.zeros(1, batch_size, self.hidden_size)
            c_0 = torch.zeros(1, batch_size, self.hidden_size)
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.label(final_hidden_state[-1])

        return final_output
