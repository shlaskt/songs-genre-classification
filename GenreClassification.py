from DataLoader import DataLoader as Loader
from torch.utils.data import DataLoader, random_split
from Lstm import Lstm
import numpy as np
import logging
import pandas as pd
import nltk
from gensim.models import Word2Vec
import torch
from SongData import SongData
import time
import torch.optim as optim
import torch.nn as nn

num_of_epochs = 1
dropout = 0.25
embedded_dim = 300
hidden_layer = 256


def load(path):
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)
    file = pd.read_csv(path)
    file = file[file['lyrics'].notnull()]
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return file, tokenizer


def word2vec(sentences, name, num_features=300, min_word_count=40, workers=4,
             context=10, downsampling=1e-3):
    model = Word2Vec(sentences,
                     workers=workers,
                     size=num_features,
                     min_count=min_word_count,
                     window=context,
                     sample=downsampling)
    model.init_sims(replace=True)
    model_name = "Dataset/{}".format(name)
    model.save(model_name)
    return model


def get_accuracy(prediction, y):
    probs = torch.softmax(prediction, dim=1)
    winners = probs.argmax(dim=1)
    correct = (winners == y.argmax(dim=1)).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, loader, optimizer, criterion, epoch):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    print(f'Epoch: {epoch + 1:02} | Starting Training...')
    for index, batch in enumerate(loader):
        optimizer.zero_grad()
        predictions = model(batch[0]).squeeze(1)
        loss = criterion(predictions, batch[1])
        acc = get_accuracy(predictions, batch[1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    print(f'Epoch: {epoch + 1:02} | Finished Training')
    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, loader, criterion, epoch):
    epoch_loss = 0
    epoch_acc = 0
    print(f'Epoch: {epoch + 1:02} | Starting Evaluation...')
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            predictions = model(batch[0]).squeeze(1)
            loss = criterion(predictions, batch[1])
            acc = get_accuracy(predictions, batch[1])
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    print(f'Epoch: {epoch + 1:02} | Finished Evaluation')
    return epoch_loss / len(loader), epoch_acc / len(loader)


def time_for_epoch(start, end):
    end_to_end = end - start
    minutes = int(end_to_end / 60)
    seconds = int(end_to_end - (minutes * 60))
    return minutes, seconds


def convert_to_one_hot(genre, vec_size):
    vec = np.zeros(vec_size)
    vec[genre] = 1
    vec = [int(val) for val in vec]
    return vec


def convert_labels_representation(labels, size):
    vectors = []
    for label in labels:
        vectors.append(convert_to_one_hot(label, size))
    return vectors


def iterate_model(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MultiLabelSoftMarginLoss()
    for epoch in range(num_of_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, epoch)
        end_time = time.time()
        epoch_mins, epoch_secs = time_for_epoch(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%')


def main():
    data, labels, label_map = Loader.parse_data_for_classification('./Dataset/lyrics15LIN.csv',
                                                                   {'Not Available', '', 'Other'}, True)
    name = 'lyrics15'
    # for training w2v model
    word2vec(data, name)
    to_open = "./Dataset/" + name
    w2v = Word2Vec.load(to_open)
    w2v.wv["<pad>"] = np.zeros(embedded_dim, )
    num_of_clusters = len(label_map)
    # replace labels to one-hot vectors
    vec_labels = convert_labels_representation(labels, num_of_clusters)
    # make dataset to tuples of (tensor(songs), tensor(vet one-hot))
    dataset = SongData(data, vec_labels, w2v)
    # create train and test set
    train_len = int(0.7 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    # create LSTM Rnn model
    lstm = Lstm(batch_size=1, output_size=num_of_clusters, hidden_size=hidden_layer,
                vocab_size=len(w2v.wv.vocab), embedding_length=embedded_dim, weights= w2v.wv.vectors)
    iterate_model(lstm, DataLoader(train_set, batch_size=1, shuffle=True),
                  DataLoader(val_set, batch_size=1, shuffle=True))


if __name__ == "__main__":
    main()
