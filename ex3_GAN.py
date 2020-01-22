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

from Networks.Generator import Generator
from Networks.Discrimnator import Discrimnator
from Networks.Gan import Gan


from train_test import GanTrainer

import matplotlib.pyplot as plt
from tensorflow.keras.backend import set_floatx

set_floatx("float32")
def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--nntype', default="AE_Network", help='The type of the network')
    # parser.add_argument('--dstype', default="num", help='The type of the dataset')

    parser.add_argument('--batches', '-bs', type=int, default=32, help='number of batches')
    parser.add_argument('--epochs', '-ep', type=int, default=20000, help='number of epochs')
    parser.add_argument('--latent_vec_size', '-z', type=int, default=100, help='The size of z of '
                                                                              'the generator')


    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')
    # parser.add_argument('--loss', default="cross_entropy", help='the loss function type')

    parser.add_argument('--plot_freq', '-pf', type=int, default=500,
                        help='iteration check point to the plot')

    parser.add_argument('--output_path', default=os.getcwd(), help='The path to keep the output')


    return parser.parse_args()


def get_network(network_type):
    import Networks
    package = Networks
    return get_object(network_type, package)


def get_optimizer(optimizer_type):
    if optimizer_type == "adam":
        return tf.keras.optimizers.Adam(1e-4)
    return None


def get_loss(loss_type, samp_num):
    loss = None

    if loss_type == "cross_entropy":
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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

def generate_zspace_interpolation(generator, latent_vec_size, output_path, interplate_images):
    fake_vec1 = np.random.normal(0, 1, (1, latent_vec_size))
    fake_vec2 = np.random.normal(0, 1, (1, latent_vec_size))

    outputs = []
    titles = []

    alphas = np.linspace(0, 1, num=interplate_images)
    for a in alphas:
        outputs.append(generator(a*fake_vec1 + (1-a)*fake_vec2, training=False))
        titles.append("a={}".format(a))

    generate_image_from_list(outputs, titles, "z_space_interpolation", output_path)



def generate_image_from_list(images, images_titles, title, output_path):

    fig, axs = plt.subplots(1, len(images), figsize=(20, 10))
    for i in range(len(images)):
        axs[i].imshow(images[i][0, :, :, 0], cmap='gray')
        axs[i].set_title(images_titles[i])
    fig.suptitle(title)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plt.savefig(os.path.join(output_path, title + ".png"))

def generate_sample(generator, latent_vec_size, output_dir):
    fake_vec = np.random.normal(0, 1, (1, latent_vec_size))
    output = generator(fake_vec, training=False)

    plt.figure()
    plt.imshow(output[0,:,:,0], cmap='gray')

    title = "Generator_output"

    plt.title(title)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(os.path.join(output_dir, title + ".png"))





def train_main(args, train_ds, plot_freq, output_path, generator,
               discriminator):

    optimizer = get_optimizer(args.optimizer)
    loss = get_loss("cross_entropy", args.batches)
    trainer = GanTrainer(generator, discriminator, optimizer, optimizer, loss, loss)

    generator_plotter = Plotter(['train'], "generator", os.path.join(output_path, "Loss"))
    discriminator_plotter = Plotter(['train'], "discriminator", os.path.join(output_path, "Loss"))

    output_for_epochs = []
    titles = []

    try:
        train_counter = 0
        train_step = trainer.get_step()


        for epoch in range(args.epochs):
            for real_images, labels in train_ds:
                fake_vecs = np.random.normal(0, 1, (args.batches, args.latent_vec_size))

                train_step(real_images, fake_vecs)

                if train_counter % plot_freq == 0:

                    template = 'Epochs {}, Discriminator Loss: {}, Generator Loss: {}'
                    print(template.format(epoch + 1,
                                          trainer.disc_loss_mean.result(),
                                          trainer.gen_loss_mean.result()))

                    discriminator_plotter.add("train", train_counter,
                                     tf.cast(trainer.disc_loss_mean.result(), tf.float32).numpy())
                    generator_plotter.add("train", train_counter,
                                     tf.cast(trainer.gen_loss_mean.result(), tf.float32).numpy())

                train_counter += 1

            trainer.disc_loss_mean.reset_states()
            trainer.gen_loss_mean.reset_states()

            # new generator image
            fake_vec = np.random.normal(0, 1, (1, args.latent_vec_size))
            output = generator(fake_vec, training=False)
            output_for_epochs.append(output)
            titles.append("epoch={}".format(epoch))

        if args.epochs > 1:
            generate_image_from_list(output_for_epochs, titles, "generator_outputs_for_epochs", output_path)


        # Reset the metrics for the next epoch
        discriminator_plotter.plot()
        generator_plotter.plot()


    finally:
        print("train is done")


def get_dataset(batches_num, *args):
    (x_train, y_train), (x_test, y_test) = get_num_dataset()

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds, test_ds = add_channel_dim(x_train, x_train, x_test, x_test, batches_num)
    return train_ds, test_ds, x_test, y_test



if __name__ == '__main__':
    args = get_args()
    tf.keras.backend.set_floatx('float64')


    train_ds, test_ds, x_test, y_test = get_dataset(args.batches)

    generator = Generator()
    discriminator = Discrimnator()

    train_main(args, train_ds, args.plot_freq,
               args.output_path, generator,discriminator)

    generate_sample(generator, args.latent_vec_size, args.output_path)
    generate_zspace_interpolation(generator, args.latent_vec_size, args.output_path, 5)









