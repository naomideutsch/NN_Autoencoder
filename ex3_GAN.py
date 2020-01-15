from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.python.keras.losses import MSE
from utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import argparse
import tensorflow as tf
import numpy as np
import os
import itertools

from Networks.Generator import Generator
from Networks.Discrimnator import Discrimnator
from Networks.Gan import Gan


from train_test import Trainer, Validator

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--nntype', default="AE_Network", help='The type of the network')
    parser.add_argument('--dstype', default="num", help='The type of the dataset')

    parser.add_argument('--batches', '-bs', type=int, default=32, help='number of batches')
    parser.add_argument('--epochs', '-ep', type=int, default=20, help='number of epochs')
    parser.add_argument('--latent_vec_size', '-z', type=int, default=10, help='The size of z of '
                                                                              'the generator')


    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')
    parser.add_argument('--loss', default="cross_entropy", help='the loss function type')

    parser.add_argument('--plot_freq', '-pf', type=int, default=1875,
                        help='iteration check point to the plot')

    parser.add_argument('--output_path', default=os.getcwd(), help='The path to keep the output')


    return parser.parse_args()


def get_network(network_type):
    import Networks
    package = Networks
    return get_object(network_type, package)


def get_optimizer(optimizer_type):
    if optimizer_type == "adam":
        return tf.keras.optimizers.Adam()
    return None


def get_loss(loss_type, samp_num):
    loss = None

    if loss_type == "cross_entropy":
        loss = BinaryCrossEntropy(samp_num)

    if loss_type == "mse":
        loss = MSE

    return loss


def add_channel_dim(train_1, train_2, test_1, test_2, batches_num):
    train_ds = tf.data.Dataset.from_tensor_slices((train_1, train_2)).shuffle(10000).batch(batches_num)
    test_ds = tf.data.Dataset.from_tensor_slices((test_1, test_2)).batch(batches_num)
    return train_ds, test_ds

def get_dataset(batches_num):
    (x_train, y_train), (x_test, y_test) = get_num_dataset()

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return x_train


def train_main(args, x_train, plot_freq, output_path, generator,
               discriminator, gan):
    half_batch = int(np.floor(args.batches/2))
    optimizer = get_optimizer(args.optimizer)
    loss = get_loss(args.loss, args.batches)

    generator_trainer = Trainer(generator, optimizer, loss)
    discriminator_trainer = Trainer(discriminator, optimizer, loss)
    gan_trainer = Trainer(gan, optimizer, loss)

    generator_plotter = Plotter(['train'], "generator", os.path.join(output_path, "Loss"))
    discriminator_plotter = Plotter(['train'], "discriminator", os.path.join(output_path, "Loss"))
    gan_plotter = Plotter(['train'], "gan", os.path.join(output_path, "Loss"))

    try:
        train_counter = 0
        generator_step = generator_trainer.get_step()
        discriminator_step = discriminator_trainer.get_step()
        gan_step = gan_trainer.get_step()

        for epoch in range(args.epochs):
            random_idx = np.rand.randint(0, len(x_train), half_batch)
            real_images = x_train[random_idx]
            fake_vecs = np.random.normal(0, 1, (half_batch, args.latent_vec_size))

            discriminator_step(real_images, np.one((half_batch, 1)))
            discriminator_step(fake_vecs, np.zeros((half_batch, 1)))

            noise = np.random.normal(0, 1, (args.batches, args.latent_vec_size))
            gan_step(noise, np.one((half_batch, 1)))

    finally:
        print("train is done")






if __name__ == '__main__':
    args = get_args()
    tf.keras.backend.set_floatx('float64')


    x_train = get_dataset(args.batches)

    generator = Generator()
    discriminator = Discrimnator()
    gan = Gan(generator, discriminator)

    train_main(args, x_train, args.plot_freq,
               args.output_path, generator,discriminator, gan)





