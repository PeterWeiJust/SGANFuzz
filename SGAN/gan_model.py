import pickle

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Lambda, Dropout, Concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils import to_categorical
tf.compat.v1.disable_eager_execution()

def Pretrain_Generator(V, E, H):

    input = Input(shape=(None,), dtype='int32', name='Input')  # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
    out = LSTM(H, return_sequences=True, name='LSTM')(out)  # (B, T, H)
    out = TimeDistributed(
        Dense(V, activation='softmax', name='DenseSoftmax'),
        name='TimeDenseSoftmax')(out)  # (B, T, V)
    generator_pretraining = Model(input, out)
    return generator_pretraining


class Generator():
    'Create Generator, which generate a next word.'

    def __init__(self, sess, B, V, E, H, lr=1e-3):
        self.sess = sess
        self.B = B
        self.V = V
        self.E = E
        self.H = H
        self.lr = lr
        self._build_gragh()
        self.reset_rnn_state()

    def _build_gragh(self):
        state_in = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        h_in = tf.compat.v1.placeholder(tf.float32, shape=(None, self.H))
        c_in = tf.compat.v1.placeholder(tf.float32, shape=(None, self.H))
        action = tf.compat.v1.placeholder(tf.float32, shape=(None, self.V))  # onehot (B, V)
        reward = tf.compat.v1.placeholder(tf.float32, shape=(None,))  # (B, )

        self.layers = []

        embedding = Embedding(self.V, self.E, mask_zero=True, name='Embedding')
        out = embedding(state_in)  # (B, 1, E) 将state_in输入embedding
        self.layers.append(embedding)

        lstm = LSTM(self.H, return_state=True, name='LSTM')
        out, next_h, next_c = lstm(out, initial_state=[h_in, c_in])  # (B, H)
        self.layers.append(lstm)

        dense = Dense(self.V, activation='softmax', name='DenseSoftmax')
        prob = dense(out)  # (B, V)
        self.layers.append(dense)

        log_prob = tf.compat.v1.log(tf.reduce_mean(prob * action, axis=-1))  # (B, )
        loss = - log_prob * reward
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        minimize = optimizer.minimize(loss)

        self.state_in = state_in
        self.h_in = h_in
        self.c_in = c_in
        self.action = action
        self.reward = reward
        self.prob = prob
        self.next_h = next_h
        self.next_c = next_c
        self.minimize = minimize
        self.loss = loss

        self.init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(self.init_op)

    def reset_rnn_state(self):
        self.h = np.zeros([self.B, self.H])
        self.c = np.zeros([self.B, self.H])

    def set_rnn_state(self, h, c):
        self.h = h
        self.c = c

    def get_rnn_state(self):
        return self.h, self.c

    def predict(self, state, stateful=True):
        # state = state.reshape(-1, 1)
        feed_dict = {
            self.state_in: state,
            self.h_in: self.h,
            self.c_in: self.c}
        prob, next_h, next_c = self.sess.run(
            [self.prob, self.next_h, self.next_c],
            feed_dict)

        if stateful:
            self.h = next_h
            self.c = next_c
            return prob
        else:
            return prob, next_h, next_c

    def update(self, state, action, reward, h=None, c=None, stateful=True):
        if h is None:
            h = self.h
        if c is None:
            c = self.c
        state = state[:, -1].reshape(-1, 1)
        reward = reward.reshape(-1)
        feed_dict = {
            self.state_in: state,
            self.h_in: h,
            self.c_in: c,
            self.action: to_categorical(action, self.V),
            self.reward: reward}
        _, loss, next_h, next_c = self.sess.run(
            [self.minimize, self.loss, self.next_h, self.next_c],
            feed_dict)

        if stateful:
            self.h = next_h
            self.c = next_c
            return loss
        else:
            return loss, next_h, next_c

    def sampling_word(self, prob):
        action = np.zeros((self.B,), dtype=np.int32)
        for i in range(self.B):
            p = prob[i]
            action[i] = np.random.choice(self.V, p=p)
        return action

    def sampling_sentence(self, T, BOS=1):
        self.reset_rnn_state()
        action = np.zeros([self.B, 1], dtype=np.int32)
        action[:, 0] = BOS
        actions = action

        for _ in range(T):
            prob = self.predict(action)

            action = self.sampling_word(prob).reshape(-1, 1)

            actions = np.concatenate([actions, action], axis=-1)
        # Remove BOS
        actions = actions[:, 1:]
        self.reset_rnn_state()
        return actions

    def generate_samples(self, T, g_data, num, output_file):
        sentences = []

        for _ in range(num // self.B + 1):
            actions = self.sampling_sentence(T)
            actions_list = actions.tolist()

            for sentence_id in actions_list:
                sentence = [g_data.id2word[action] for action in sentence_id if action != 0 and action != 2]
                sentences.append(sentence)

        output_str = ''

        for i in range(num):
            output_str += ' '.join(sentences[i]) + '\n'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)

    def generate_single_seq(self, T, g_data):
        sentences = []

        actions = self.sampling_sentence(T)
        actions_list = actions.tolist()

        for sentence_id in actions_list:
            sentence = [g_data.id2word[action] for action in sentence_id if action != 0 and action != 2]
            sentences.append(sentence)

        output_str = ''
        output_str += ' '.join(sentences[0])
        return output_str
    def generate_single_payload(self, T, g_data):
        sentences = []

        actions = self.sampling_sentence(T)
        actions_list = actions.tolist()

        for sentence_id in actions_list:
            sentence = [g_data.id2word[action] for action in sentence_id if action != 0 and action != 2]
            sentences.append(sentence)

        output_str = ''
        output_str += ''.join(sentences[0])
        return output_str

    def save(self, path):
        weights = []
        for layer in self.layers:
            w = layer.get_weights()
            weights.append(w)
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)


def Discriminator(V, E, H=64, dropout=0.1):
    input = Input(shape=(None,), dtype='int32', name='Input')  # (B, T)
    out = Embedding(V, E, mask_zero=True, name='Embedding')(input)  # (B, T, E)
    out = LSTM(H)(out)
    out = Highway(out, num_layers=1)
    out = Dropout(dropout, name='Dropout')(out)
    out = Dense(1, activation='sigmoid', name='FC')(out)

    discriminator = Model(input, out)
    return discriminator

def Highway(x, num_layers=1, activation='relu', name_prefix=''):
    input_size = K.int_shape(x)[1]
    for i in range(num_layers):
        gate_ratio_name = '{}Highway/Gate_ratio_{}'.format(name_prefix, i)
        fc_name = '{}Highway/FC_{}'.format(name_prefix, i)
        gate_name = '{}Highway/Gate_{}'.format(name_prefix, i)

        gate_ratio = Dense(input_size, activation='sigmoid', name=gate_ratio_name)(x)
        fc = Dense(input_size, activation=activation, name=fc_name)(x)
        x = Lambda(lambda args: args[0] * args[2] + args[1] * (1 - args[2]), name=gate_name)([fc, x, gate_ratio])
    return x
