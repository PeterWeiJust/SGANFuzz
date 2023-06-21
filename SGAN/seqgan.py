import os

import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.optimizers import Adam

from SGAN.gan_model import Pretrain_Generator, Discriminator
from SGAN.reinforcement_learning import Agent, Environment
from SGAN.function_helper import Pretrain_Generator_Sequence, Discriminator_Sequence

sess = tf.compat.v1.Session()
tf.compat.v1.keras.backend.set_session(sess)


class SeqGAN(object):

    def __init__(self, B, T, g_E, g_H, d_E, d_H, d_dropout, path_pos, path_neg, g_lr=1e-3, d_lr=1e-3, n_sample=16,
                 generate_samples=10000, init_eps=0.1):
        self.B, self.T = B, T
        self.g_E, self.g_H = g_E, g_H
        self.d_E, self.d_H = d_E, d_H
        self.d_dropout = d_dropout
        self.generate_samples = generate_samples
        self.g_lr, self.d_lr = g_lr, d_lr
        self.eps = init_eps
        self.init_eps = init_eps
        self.top = os.getcwd()
        self.path_pos = path_pos
        self.path_neg = path_neg

        self.g_data = Pretrain_Generator_Sequence(self.path_pos, B=B, T=T,
                                                  min_count=1)
        self.d_data = Discriminator_Sequence(path_pos=self.path_pos, path_neg=self.path_neg, B=self.B,
                                             shuffle=True)

        self.V = self.g_data.V
        self.agent = Agent(sess, B, self.V, g_E, g_H, g_lr)
        self.g_beta = Agent(sess, B, self.V, g_E, g_H, g_lr)

        self.discriminator = Discriminator(self.V, d_E, d_H, d_dropout)

        self.env = Environment(self.discriminator, self.g_data, self.g_beta, n_sample=n_sample)

        self.generator_pre = Pretrain_Generator(self.V, g_E, g_H)

    def pre_train(self, g_epochs=3, d_epochs=1, g_pre_path=None, d_pre_path=None, g_lr=1e-3, d_lr=1e-3):
        self.pre_train_generator(g_epochs=g_epochs, g_pre_path=g_pre_path, lr=g_lr)

        self.pre_train_discriminator(d_epochs=d_epochs, d_pre_path=d_pre_path, lr=d_lr)

    def pre_train_generator(self, g_epochs=3, g_pre_path=None, lr=1e-3):
        if g_pre_path is None:
            self.g_pre_path = os.path.join(self.top, 'data', 'save', 'generator_pre.h5')
        else:
            self.g_pre_path = g_pre_path

        g_adam = Adam(lr)
        self.generator_pre.compile(g_adam, 'categorical_crossentropy')
        print('Generator pre-training')
        self.generator_pre.summary()

        pre_gen = self.generator_pre.fit(
            self.g_data,
            steps_per_epoch=None,
            epochs=g_epochs)
        pre_gen_loss = self.np_loss(pre_gen, "pre_gen")
        np.save("pre_gen_loss.npy", pre_gen_loss)

        self.generator_pre.save(self.g_pre_path)
        self.reflect_pre_train()

    def pre_train_discriminator(self, d_epochs=1, d_pre_path=None, lr=1e-3):
        if d_pre_path is None:
            self.d_pre_path = os.path.join(self.top, 'data', 'save', 'discriminator_pre.h5')
        else:
            self.d_pre_path = d_pre_path

        print('Discriminator pre-training')
        self.agent.generator.generate_samples(self.T, self.g_data,
                                              self.generate_samples, self.path_neg)

        self.d_data = Discriminator_Sequence(
            path_pos=self.path_pos,
            path_neg=self.path_neg,
            B=self.B,
            shuffle=True)

        d_adam = Adam(lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.discriminator.summary()

        pre_dis = self.discriminator.fit(
            self.d_data,
            steps_per_epoch=None,
            epochs=d_epochs)
        pre_dis_loss = self.np_loss(pre_dis, "pre_gen")
        np.save("pre_dis_loss.npy", pre_dis_loss)

        self.discriminator.save(self.d_pre_path)

    def load_pre_train(self, g_pre_path, d_pre_path):
        self.generator_pre = load_model(g_pre_path)
        self.reflect_pre_train()
        self.discriminator = load_model(d_pre_path)

    def reflect_pre_train(self):
        i = 0
        for layer in self.generator_pre.layers:
            if len(layer.get_weights()) != 0:
                w = layer.get_weights()
                self.agent.generator.layers[i].set_weights(w)
                self.g_beta.generator.layers[i].set_weights(w)
                i += 1
        print("reflect pre train finished")

    def train(self, steps=200, g_steps=1, d_steps=5, d_epochs=1,
              g_weights_path='data/save/generator.pkl',
              d_weights_path='data/save/discriminator.h5',
              verbose=True,
              head=1):
        d_adam = Adam(self.d_lr)
        self.discriminator.compile(d_adam, 'binary_crossentropy')
        self.eps = self.init_eps
        gan_generator_loss = []
        gan_discriminator_loss = []
        for step in range(steps):
            # Generator training
            print("Epoch: ", step)
            loss = []
            for _ in range(g_steps):
                print("Epoch: {}, Generator step: {}".format(step, g_steps))
                rewards = np.zeros([self.B, self.T])
                self.agent.reset()
                self.env.reset()
                sen_loss = []

                for t in range(self.T):
                    state = self.env.get_state()

                    action = self.agent.act(state, epsilon=0.0)

                    _next_state, reward, is_episode_end, _info = self.env.step(action)
                    gen_loss = self.agent.generator.update(state, action, reward)
                    sen_loss.append(gen_loss)
                    rewards[:, t] = reward.reshape([self.B, ])
                    if is_episode_end:
                        if verbose:
                            print('Reward: {:.3f}, Episode end'.format(np.average(rewards)))
                            self.env.render(head=head)
                        break

                loss.append(np.mean(sen_loss))
                print("gen loss : {}".format(np.mean(sen_loss)))

            np.save("gan_gen_seq_loss.npy", loss)
            print("Epoch: {}, Generator loss: {}".format(step, np.mean(loss)))
            gan_generator_loss.append(np.mean(loss))

            loss_dis = []
            # Discriminator training
            for _ in range(d_steps):
                self.agent.generator.generate_samples(
                    self.T,
                    self.g_data,
                    self.generate_samples,
                    self.path_neg)
                self.d_data = Discriminator_Sequence(
                    path_pos=self.path_pos,
                    path_neg=self.path_neg,
                    B=self.B,
                    shuffle=True)
                gan_dis = self.discriminator.fit(
                    self.d_data,
                    steps_per_epoch=None,
                    epochs=d_epochs)
                loss_dis.append(self.np_loss(gan_dis, "gan_dis"))

            gan_discriminator_loss.append(np.mean(loss_dis))
            # Update env.g_beta to agent
            self.agent.save(g_weights_path)
            self.g_beta.load(g_weights_path)

            self.discriminator.save(d_weights_path)
            self.eps = max(self.eps * (1 - float(step) / steps * 4), 1e-4)

        np.save("gan_dis_loss.npy", gan_discriminator_loss)
        np.save("gan_gen_loss.npy", gan_generator_loss)

    def np_loss(self, model, name):
        lossy = model.history['loss']
        np_lossy = np.array(lossy).reshape((1, len(lossy)))
        return np_lossy

    def save(self, g_path, d_path):
        self.agent.save(g_path)
        self.discriminator.save(d_path)

    def load(self, g_path, d_path):
        self.agent.load(g_path)
        self.g_beta.load(g_path)
        self.discriminator = load_model(d_path)

    def generate_batch_text(self, file_name, generate_samples):
        path_neg = os.path.join(self.top, 'data', 'save', file_name)

        self.agent.generator.generate_samples(
            self.T, self.g_data, generate_samples, path_neg)

