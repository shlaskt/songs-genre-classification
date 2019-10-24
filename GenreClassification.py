from DataLoader import DataLoader
from Lstm import Lstm
import random
import numpy as np
import gensim
import logging
import pandas as pd
import nltk
from gensim.models import Word2Vec
import torch


def shuffle_data(x, y):
    zip_x_y = list(zip(x, y))
    random.shuffle(zip_x_y)
    new_x, new_y = zip(*zip_x_y)
    return new_x, new_y


def split_data(data, labels):
    cut = int(len(data) * 0.8)
    x = data[: cut]
    y = labels[: cut]
    test = data[cut:]
    test_y = labels[cut:]
    return x, y, test, test_y


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


def validation(lstm, data, labels, idx2genre):
    """
    Used to test the model on the validation set, prints the model's accuracy.
    :param conv: the model.
    :param valid_set: the validation set to predict.
    :param device: CPU or GPU.
    :return: null
    """
    lstm.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for song, genre in zip(data, labels):
            # calc the prediction vector using the model's forward pass.
            pred_vec = lstm(song)
            # get the item with highest probability.
            _, pred = torch.max(pred_vec.data, 1)
            # sum the sample and correct to calculate the ratio.
            total += 1
            print("song: "+str(total)+" pred: "+idx2genre[pred]+', correct: '+idx2genre[genre])
            correct += (pred == genre).sum().item()
        # print the accuracy of the model.
        print('Test Accuracy of the model: {}%'.format((correct / total) * 100))


def train(lstm, opt, data, labels):
    """
    Trains the model on the train set, using the forward pass of the model,
    the Cross Entropy Loss (which is NLL + softmax), using Adam as the optimizer.
    :param train_set: the set to train to model with.
    :param conv: the model.
    :param opt: the optimizer (Adam)
    :param lr_decay: the rate of decay of the learning rate.
    :param device: CPU or GPU
    :return: conv (the trained model)
    """
    loss_function = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        lstm.train()
        # print('Epoch: ' + str(epoch))
        for k, (song, genre) in enumerate(zip(data, labels)):
            prediction = lstm(song)
            label = np.zeros(10)
            label[genre] = 1
            loss = loss_function(prediction, torch.from_numpy(label))
            opt.zero_grad()
            loss.backward()
            opt.step()


def main():
    data, labels, label_map = DataLoader.parse_data_for_classification('./Dataset/lyrics15LIN.csv',
                                                    {'Not Available', '', 'zora sourit', 'Alkebulan', 'Other'})
    data, labels = shuffle_data(data, labels)
    name = 'lyrics15'
    # word2vec(data, name)
    toOpen = "./Dataset/" + name
    model = Word2Vec.load(toOpen)
    data, labels, valid_data, valid_labels = split_data(data, labels)
    lstm = Lstm(model)
    opt = torch.optim.Adam(lstm.parameters(), lr=0.01)
    train(lstm, opt, data, labels)
    validation(lstm, valid_data, valid_labels, label_map)


if __name__ == "__main__":
    main()
