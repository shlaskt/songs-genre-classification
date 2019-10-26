import pandas as pd
import numpy as np


def construct_dict_encoders(unique_items):
    item2idx = dict((item, index) for index, item in enumerate(unique_items))
    idx2item = dict((index, item) for index, item in enumerate(unique_items))
    return item2idx, idx2item


def cleaner(file, to_replace, genre_filter, arg):
    df = pd.read_csv(file, delimiter=',')
    df = df[df['lyrics'] != 'instrumental'].dropna()
    df = df[df['lyrics'].map(len) > 1]
    df = genre_filter(df, arg)
    data = df[['lyrics', 'genre']]
    songs = data.sample(len(data))
    songs['lyrics'] = songs['lyrics'].str.strip('[]')
    songs['lyrics'] = songs['lyrics'].str.strip('()')
    for token in to_replace:
        songs['lyrics'] = songs['lyrics'].str.replace(token, '')
    return songs


def convert_to_one_hot(val, vec_size):
    vec = np.zeros(vec_size)
    vec[val] = 1
    vec = [int(val) for val in vec]
    return vec


def ignorer(df, to_ignore):
    for key in to_ignore:
        df = df[df['genre'] != key]
    return df


def genre_selector(df, genre):
    df = df[df['genre'] == genre]
    return df


class DataUtils:
    @staticmethod
    def parse_data_for_classification(file, to_ignore, is_limit=False):
        data = cleaner(file, {'chorus', '[^\w\s]', ':', ',', 'verse',
                              'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}, ignorer, to_ignore)
        data = data.replace({'\n': ' '}, regex=True)
        data.index = pd.RangeIndex(len(data.index))
        songs = []
        labels = data['genre'].tolist()
        lyrics = data['lyrics'].tolist()
        gen2idx, idx2gen = construct_dict_encoders(data['genre'].dropna().drop_duplicates().tolist())
        labels = [gen2idx[label] for label in labels]
        for row in lyrics:
            song = []
            for word in row.split(' '):
                song.append(word)
            if is_limit:
                if len(song) < 200:
                    song += (["<pad>"] * (200 - len(song)))
                elif len(song) > 200:
                    song = song[:200]
            songs.append(song)

        return songs, labels, idx2gen

    @staticmethod
    def convert_representation(data, vec_size):
        vectors = []
        for val in data:
            vectors.append(convert_to_one_hot(val, vec_size))
        return vectors

    @staticmethod
    def character_encoding(file, genre, max_vec_len, step):
        to_replace = {'[^\w\s]', ':', ',', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        songs = cleaner(file, to_replace, genre_selector, genre)
        text = ''
        for row in songs['lyrics']:
            text = text + str(row).lower()
        unique_chars = sorted(list(set(text)))
        char2idx, idx2char = construct_dict_encoders(unique_chars)
        sentences = []
        next_chars = []
        for i in range(0, len(text) - max_vec_len, step):
            sentences.append(text[i: i + max_vec_len])
            next_chars.append(text[i + max_vec_len])
        data = np.zeros((len(sentences), max_vec_len, len(unique_chars)), dtype=np.bool)
        labels = np.zeros((len(sentences), len(unique_chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                data[i, t, char2idx[char]] = 1
            labels[i, char2idx[next_chars[i]]] = 1
        return data, labels, idx2char, unique_chars, char2idx

    @staticmethod
    def translator(unique_chars, seed, char2idx):
        parsed = np.zeros((len(seed), len(unique_chars)), dtype=np.bool)
        for i, char in enumerate(seed):
            parsed[i, char2idx[char]] = 1
        return parsed


