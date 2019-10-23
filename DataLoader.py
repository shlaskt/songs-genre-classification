import pandas as pd


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
    def parse_data(file, to_ignore):
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
        gen2idx, idx2gen = construct_map(info['genre'].dropna().drop_duplicates().tolist())
        labels = [gen2idx[label] for label in labels]
        for row in info['lyrics']:
            data_set.append(row)
        return data_set, labels, idx2gen

