from DataUtils import DataUtils
from keras.models import Sequential
from keras.layers import LSTM, Activation
from keras.layers import Dense
from keras import optimizers
import numpy as np

max_vec_len = 20
step = 1
epochs = 20
word_count = 600


def predict(model, seed, char2idx, idx2char, unique_chars):
    pattern = DataUtils.translator(unique_chars, seed, char2idx)
    res = '' + seed
    for i in range(word_count):
        x = np.reshape(pattern, (1, len(pattern), len(unique_chars)))
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = idx2char[index]
        res += result
        seq = np.zeros((1, len(unique_chars)), dtype=bool)
        seq[0, index] = 1
        pattern = np.concatenate((pattern, seq))
        pattern = pattern[1:]
    print(res)


def main():
    data, labels, idx2char, unique_chars, char2idx = DataUtils.character_encoding('./Dataset/lyrics15LIN.csv',
                                                                                  'Country', max_vec_len, step)
    num_of_chars = len(unique_chars)
    model = Sequential()
    model.add(LSTM(128, input_shape=(max_vec_len, num_of_chars)))
    model.add(Dense(num_of_chars))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001))
    model.fit(data, labels, batch_size=128, epochs=epochs)
    model.save('./Dataset/15k-30epoch')
    predict(model, 'country road take me', char2idx, idx2char, unique_chars)


if __name__ == "__main__":
    main()
