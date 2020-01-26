from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow.python.keras.losses import MSE, BinaryCrossentropy

from utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import argparse
import tensorflow as tf
import numpy as np
import os
import itertools

from Networks.Decoder import Decoder

from train_test import GloTrainer

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batches', '-bs', type=int, default=32, help='number of batches')
    parser.add_argument('--epochs', '-ep', type=int, default=1, help='number of epochs')
    parser.add_argument('--latent_vec_size', '-z', type=int, default=100, help='The size of z of '
                                                                               'the generator')
    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')
    parser.add_argument('--plot_freq', '-pf', type=int, default=500,
                        help='iteration check point to the plot')
    parser.add_argument('--output_path', default=os.getcwd(), help='The path to keep the output')
    return parser.parse_args()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Creates Training Objects By Names ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_network(network_type):
    import Networks
    package = Networks
    return get_object(network_type, package)


def get_optimizer(optimizer_type):
    if optimizer_type == "adam":
        return tf.keras.optimizers.Adam(1e-4)
    return None


def get_loss(loss_type):
    loss = None
    if loss_type == "MSE":
        loss = tf.keras.losses.MSE
    return loss


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data Loaders ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_dataset(batch_size, latent_vec_size):
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = x_train.reshape(x_train.shape[0], 28, 28, 1)

    train_images = train_images / 255.0  # Normalize the images to [-1, 1]
    train_ds = tf.data.Dataset.from_tensor_slices(train_images).shuffle(x_train.shape[0]).batch(batch_size)
    return train_ds, x_train.shape[0]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Output functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#




def generate_sample(model, latent_vec_size, output_dir):
    fake_vec = tf.Variable(np.random.normal(size=(1, latent_vec_size)), trainable=False).numpy()
    output = model(fake_vec, training=False)

    plt.figure()
    plt.imshow(output[0, :, :, 0] * 255.0, cmap='gray')

    title = "GLO_output"

    plt.title(title)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, title + ".png"))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#



def train_main(args, real_ds, ds_size, plot_freq, output_path, model):

    model_optimizer = get_optimizer(args.optimizer)
    z_space_optimizer = get_optimizer(args.optimizer)

    loss = get_loss("MSE")
    trainer = GloTrainer(model, model_optimizer, z_space_optimizer, loss, ds_size, args.latent_vec_size)

    plotter = Plotter(['model loss', 'z space Loss'], "GLO", os.path.join(output_path, "Loss"))

    z_space_vecs = tf.Variable(np.random.normal(size=(ds_size, args.latent_vec_size))).numpy()

    try:
        batch_idx = 0
        train_counter = 0
        train_step = trainer.get_step()

        for epoch in range(args.epochs):
            for real_images in real_ds:
                train_counter += 1
                relevant_z_vecs = tf.Variable(z_space_vecs[batch_idx: batch_idx + real_images.shape[0]], trainable=True)

                train_step(real_images, relevant_z_vecs)
                z_space_vecs[batch_idx: batch_idx + real_images.shape[0]] = relevant_z_vecs.numpy()


                if train_counter % plot_freq == 0:

                    template = 'Epochs {}, model Loss: {}, z space Loss: {}'
                    print(template.format(epoch + 1,
                                          trainer.model_loss_mean.result(), trainer.z_space_loss_mean.result()))

                    plotter.add("model loss", train_counter,
                                              tf.cast(trainer.model_loss_mean.result(), tf.float32).numpy())
                    plotter.add("z space Loss", train_counter,
                                              tf.cast(trainer.z_space_loss_mean.result(), tf.float32).numpy())

                train_counter += 1
                batch_idx += real_images.shape[0]

            trainer.model_loss_mean.reset_states()
            trainer.z_space_loss_mean.reset_states()

            batch_idx = 0

        # Reset the metrics for the next epoch
        plotter.plot()


    finally:
        print("train is done")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


if __name__ == '__main__':
    args = get_args()
    tf.keras.backend.set_floatx('float64')

    train_ds, dataset_size = get_dataset(args.batches, args.latent_vec_size)
    model = Decoder()


    train_main(args, train_ds, dataset_size, args.plot_freq,
               args.output_path, model)
    generate_sample(model, args.latent_vec_size, args.output_path)










