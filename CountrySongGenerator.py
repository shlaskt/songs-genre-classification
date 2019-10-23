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


def predict(model, seed, char2idx, idx2char, unique_chars):
    pattern = DataLoader.translator(unique_chars, seed, char2idx)
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), len(unique_chars)))
        # x = x / float(len(unique_chars))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = idx2char[index]
        # seq_in = [idx2char[value] for value in pattern]
        print(result)
        seq = np.zeros((1, len(unique_chars)), dtype=bool)
        seq[0, index] = 1
        pattern = np.concatenate((pattern, seq))
        pattern = pattern[1:]


def main():
    data, labels, idx2char, unique_chars, char2idx = DataLoader.character_encoding('./Dataset/lyrics.csv', 'Country',
                                         Constants.max_vec_len.value, Constants.step.value)
    num_of_chars = len(unique_chars)
    model = Sequential()
    model.add(LSTM(128, input_shape=(Constants.max_vec_len.value, num_of_chars)))
    model.add(Dense(num_of_chars))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.01))
    model.fit(data, labels, batch_size=128, epochs=1)
    predict(model, 'sweet home alabama i', char2idx, idx2char, unique_chars)
    print('hello')


if __name__ == "__main__":
    main()
