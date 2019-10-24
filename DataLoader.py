import pandas as pd
import numpy as np


def construct_map(genres):
    gen2idx = {}
    idx2gen = {}
    index = 0
    for genre in genres:
        gen2idx[genre] = index
        idx2gen[index] = genre
        index += 1
    return gen2idx, idx2gen


class DataLoader:
    @staticmethod
    def parse_data_for_classification(file, to_ignore, is_limit = False):
        lyrics = pd.read_csv(file, delimiter=',')
        lyrics = lyrics[lyrics['lyrics'] != 'instrumental'].dropna()
        lyrics = lyrics[lyrics['lyrics'].map(len) > 1]
        for key in to_ignore:
            lyrics = lyrics[lyrics['genre'] != key]
        lyrics.dataframeName = 'lyrics.csv'
        info = lyrics[['lyrics', 'genre']]
        info = info.sample(len(info))
        to_replace = {'chorus', '[^\w\s]', ':', ',', 'verse', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'}
        info['lyrics'] = info['lyrics'].str.lower()
        info['lyrics'] = info['lyrics'].str.strip('[]')
        info['lyrics'] = info['lyrics'].str.strip('()')
        for token in to_replace:
            info['lyrics'] = info['lyrics'].str.replace(token, '')
        info = info.replace({'\n': ' '}, regex=True)
        info.index = pd.RangeIndex(len(info.index))
        data_set = []
        labels = info['genre'].tolist()
        data = info['lyrics'].tolist()
        gen2idx, idx2gen = construct_map(info['genre'].dropna().drop_duplicates().tolist())
        labels = [gen2idx[label] for label in labels]
        for row in data:
            song = []
            for word in row.split(' '):
                song.append(word)
            if (is_limit):
                if len(song) < 200:
                    song += (["<pad>"] * (200 - len(song)))
                elif len(song) > 200:
                    song = song[:200]
            data_set.append(song)

        return data_set, labels, idx2gen

    @staticmethod
    def character_encoding(file, genre, max_vec_len, step):
        songs = pd.read_csv(file, delimiter=',')
        songs = songs[songs['genre'] == genre]
        songs = songs[['lyrics', 'genre']]
        to_replace = {'[^\w\s]', ':', ',', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        songs['lyrics'] = songs['lyrics'].str.strip('[]')
        songs['lyrics'] = songs['lyrics'].str.strip('()')
        for token in to_replace:
            songs['lyrics'] = songs['lyrics'].str.replace(token, '')
        text = ''
        for row in songs['lyrics']:
            text = text + str(row).lower()
        unique_chars = sorted(list(set(text)))
        char2idx = dict((c, i) for i, c in enumerate(unique_chars))
        idx2char = dict((i, c) for i, c in enumerate(unique_chars))
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
        print(unique_chars)
        return data, labels, idx2char, unique_chars, char2idx

    @staticmethod
    def translator(unique_chars, seed, char2idx):
        parsed = np.zeros((len(seed), len(unique_chars)), dtype=np.bool)
        for i, char in enumerate(seed):
            parsed[i, char2idx[char]] = 1
        return parsed


