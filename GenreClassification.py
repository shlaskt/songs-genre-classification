from DataLoader import DataLoader
import random
import torch
import torch.nn as nn
import numpy as np



learning_rate = 2e-5
batch_size = 32
hidden_size = 256
embedding_length = 300


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


def train(lyrics, labels, lstm, opt):
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
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(10):
        lstm.train()
        # print('Epoch: ' + str(epoch))
        for k, (input, label) in enumerate(zip(lyrics, labels)):
            prediction = lstm(input)
            print(prediction)
            loss = loss_function(prediction, label)
            opt.zero_grad()
            loss.backward()
            opt.step()


def main():
    data, labels, label_map = DataLoader.parse_data('./Dataset/lyrics15LIN.csv',
                                                    {'Not Available', '', 'zora sourit', 'Alkebulan', 'Other'})
    data, labels = shuffle_data(data, labels)
    data, labels, valid_data, valid_labels = split_data(data, labels)


if __name__ == "__main__":
    main()
