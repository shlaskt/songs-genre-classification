from DataLoader import DataLoader
import random
import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, LSTM, Dropout, GlobalMaxPooling1D, Activation
from keras.layers import Dense
from keras import optimizers
import numpy as np
from enum import Enum


class Constants(Enum):
    max_vec_len = 20
    step = 1


def predict(model, seed, char2idx, idx2char):
    parsed = DataLoader.translator(seed, char2idx)


def main():
    data, labels, idx2char, unique_chars, char2idx = DataLoader.character_encoding('./Dataset/lyrics15LIN.csv', 'Country',
                                         Constants.max_vec_len.value, Constants.step.value)
    num_of_chars = len(unique_chars)
    model = Sequential()
    model.add(LSTM(128, input_shape=(Constants.max_vec_len.value, num_of_chars)))
    model.add(Dense(num_of_chars))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.01))
    model.fit(data, labels, batch_size=128, epochs=1)
    predict(model, '', char2idx, idx2char)
    print('hello')


if __name__ == "__main__":
    main()
