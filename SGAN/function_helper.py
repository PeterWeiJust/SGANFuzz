import numpy as np
import random
import linecache
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

import code

class Vocab:
    def __init__(self, word2id, unk_token):
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.unk_token = unk_token

    def build_vocab(self, sentences, min_count=1):
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

        self.raw_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter}

    def sentence_to_ids(self, sentence):
        return [self.word2id[word] if word in self.word2id else self.word2id[self.unk_token] for word in sentence]

def load_data(file_path):
    data = []
    for line in open(file_path, encoding='utf-8'):
        words = line.strip().split()
        data.append(words)

    return data

def sentence_to_ids(vocab, sentence, UNK=3):
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    return ids

def pad_seq(seq, max_length, PAD=0):
    seq += [PAD for i in range(max_length - len(seq))]
    return seq

def print_ids(ids, vocab, verbose=True, exclude_mark=True, PAD=0, BOS=1, EOS=2):
    sentence = []
    for i, id in enumerate(ids):
        word = vocab.id2word[id]
        if exclude_mark and id == EOS:
            break
        if exclude_mark and id in (BOS, PAD):
            continue
        sentence.append(sentence)
    if verbose:
        print(sentence)
    return sentence


class Pretrain_Generator_Sequence(Sequence):
    def __init__(self, path, B, T=40, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.path = path
        self.B = B
        self.T = T
        self.min_count = min_count

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)
        sentences = load_data(path)
        self.vocab.build_vocab(sentences, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)
        with open(path, 'r', encoding='utf-8') as f:
            self.n_data = sum(1 for line in f)
        
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        
        self.reset()


    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        x, y_true = [], []
        start = idx * self.B + 1
        end = (idx + 1) * self.B + 1
        max_length = 0
        for i in range(start, end):
            #if self.shuffle:
                #idx = self.shuffled_indices[i]
            #else:
            idx = i
            sentence = linecache.getline(self.path, idx)
            words = sentence.strip().split()
            ids = sentence_to_ids(self.vocab, words)

            ids_x, ids_y_true = [], []

            ids_x.append(self.BOS)
            ids_x.extend(ids)
            ids_x.append(self.EOS) # ex. [BOS, 8, 10, 6, 3, EOS]
            x.append(ids_x)

            ids_y_true.extend(ids)
            ids_y_true.append(self.EOS) # ex. [8, 10, 6, 3, EOS]
            y_true.append(ids_y_true)

            max_length = max(max_length, len(ids_x))


        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(x):
            x[i] = x[i][:max_length]
        for i, ids in enumerate(y_true):
            y_true[i] = y_true[i][:max_length]

        x = [pad_seq(sen, max_length) for sen in x]
        x = np.array(x, dtype=np.int32)

        y_true = [pad_seq(sen, max_length) for sen in y_true]
        y_true = np.array(y_true, dtype=np.int32)

        y_true = to_categorical(y_true, num_classes=self.V)

        return (x, y_true)

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        x, y_true = self.__getitem__(self.idx)
        self.idx += 1
        return (x, y_true)

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.shuffled_indices = np.arange(self.n_data)
            random.shuffle(self.shuffled_indices)

    def on_epoch_end(self):
        self.reset()
        pass

    def __iter__(self):
        return self

class Discriminator_Sequence(Sequence):
    def __init__(self, path_pos, path_neg, B, T=40, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.path_pos = path_pos
        self.path_neg = path_neg
        self.B = B
        self.T = T
        self.min_count = min_count

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)
        sentences = load_data(path_pos)
        self.vocab.build_vocab(sentences, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)
        with open(path_pos, 'r', encoding='utf-8') as f:
            self.n_data_pos = sum(1 for line in f)
        with open(path_neg, 'r', encoding='utf-8') as f:
            self.n_data_neg = sum(1 for line in f)

        self.n_data = self.n_data_pos + self.n_data_neg
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        self.reset()

    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        X, Y = [], []
        start = idx * self.B + 1
        end = (idx + 1) * self.B + 1
        max_length = 0
        for i in range(start, end):
            idx = self.indicies[i]
            is_pos = 1
            if idx < 0:
                is_pos = 0
                idx = -1 * idx
            idx = idx - 1

            if is_pos == 1:
                sentence = linecache.getline(self.path_pos, idx) # str
            elif is_pos == 0:
                sentence = linecache.getline(self.path_neg, idx) # str

            words = sentence.strip().split()  # list of str
            ids = sentence_to_ids(self.vocab, words) # list of ids

            x = []
            x.extend(ids)
            x.append(self.EOS) # ex. [8, 10, 6, 3, EOS]
            X.append(x)
            Y.append(is_pos)

            max_length = max(max_length, len(x))

        if self.T is not None:
            max_length = self.T

        for i, ids in enumerate(X):
            X[i] = X[i][:max_length]

        X = [pad_seq(sen, max_length) for sen in X]
        X = np.array(X, dtype=np.int32)

        return (X, Y)

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        X, Y = self.__getitem__(self.idx)
        self.idx += 1
        return (X, Y)

    def reset(self):
        self.idx = 0
        pos_indices = np.arange(start=1, stop=self.n_data_pos+1)
        neg_indices = -1 * np.arange(start=1, stop=self.n_data_neg+1)
        self.indicies = np.concatenate([pos_indices, neg_indices])
        if self.shuffle:
            random.shuffle(self.indicies)

    def on_epoch_end(self):
        self.reset()
        pass

    def __iter__(self):
        return self
