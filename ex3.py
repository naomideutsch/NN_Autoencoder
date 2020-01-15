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

from train_test import Trainer, Validator

import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--nntype', default="AE_Network", help='The type of the network')
    parser.add_argument('--dstype', default="num", help='The type of the dataset (num/denoise)')

    parser.add_argument('--batches', '-bs', type=int, default=32, help='number of batches')
    parser.add_argument('--epochs', '-ep', type=int, default=20, help='number of epochs')
    parser.add_argument('--optimizer', '-opt', default="adam", help='optimizer  type')
    parser.add_argument('--ts', type=int, default=None, help='train size')
    parser.add_argument('--loss', default="cross_entropy", help='the loss function type')

    parser.add_argument('--plot_freq', '-pf', type=int, default=1875,
                        help='iteration check point to the plot')

    parser.add_argument('--output_path', default=os.getcwd(), help='The path to keep the output')

    parser.add_argument('--max_visualization', default=2000, type=int, help='number of samples to visualize')
    parser.add_argument('--embed_tech', default="lda", help='lda/tsne')
    parser.add_argument('--percent', default=0.2, type=float, help='percent of noise in image')
    parser.add_argument('--reg_flag', default=False, action='store_true', help='add reg or not')
    parser.add_argument('--reg_num', default=0, type=int, help='regularization num')

    return parser.parse_args()


def get_network(network_type):
    import Networks
    package = Networks
    return get_object(network_type, package)


def get_optimizer(optimizer_type):
    if optimizer_type == "adam":
        return tf.keras.optimizers.Adam()
    return None


def get_loss(loss_type, sump_num, with_reg=False, alpha=0):
    loss = None
    if loss_type == "cross_entropy":
        loss = BinaryCrossEntropy(sump_num)

    if loss_type == "mse":
        loss = MSE

    if with_reg:
        loss = add_density_regularization(loss, alpha)

    return loss, with_reg


def add_channel_dim(train_1, train_2, test_1, test_2, batches_num):

    train_ds = tf.data.Dataset.from_tensor_slices((train_1, train_2)).shuffle(10000).batch(batches_num)
    test_ds = tf.data.Dataset.from_tensor_slices((test_1, test_2)).batch(batches_num)
    return train_ds, test_ds


def get_denoise_dataset(batches_num, p=0.2):
    (x_train, y_train), (x_test, y_test) = get_num_dataset()
    x_denoise = get_denoising_dataset([x_train, x_test], p)
    x_train_noise, x_test_noise = x_denoise[0], x_denoise[1]

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    x_train_noise = x_train_noise[..., tf.newaxis]
    x_test_noise = x_test_noise[..., tf.newaxis]

    # # Add a channels dimension
    # train_ds = tf.data.Dataset.from_tensor_slices(
    #     (x_train, x_train_noise)).shuffle(10000).batch(batches_num)
    # test_ds = tf.data.Dataset.from_tensor_slices((x_test, x_test_noise)).batch(batches_num)

    train_ds, test_ds = add_channel_dim(x_train, x_train_noise, x_test, x_test_noise, batches_num)
    return train_ds, test_ds, x_test_noise, y_test


def get_dataset(batches_num, *args):
    (x_train, y_train), (x_test, y_test) = get_num_dataset()

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # # Add a channels dimension
    # train_ds = tf.data.Dataset.from_tensor_slices(
    #     (x_train, x_train)).shuffle(10000).batch(batches_num)
    # test_ds = tf.data.Dataset.from_tensor_slices((x_test, x_test)).batch(batches_num)
    train_ds, test_ds = add_channel_dim(x_train, x_train, x_test, x_test, batches_num)
    return train_ds, test_ds, x_test, y_test


def train_main(epochs, train_ds, test_ds, trainer, validator, plot_freq, network_type, output_path):

    loss_plotter = Plotter(['train'], network_type, os.path.join(output_path, "Loss"))
    try:
        train_counter = 0
        train_step = trainer.get_step()
        test_step = validator.get_step()
        for epoch in range(epochs):
            for images, labels in train_ds:
                train_step(images, labels)
                train_counter += 1
                if train_counter % plot_freq == 0:
                    for test_images, test_labels in test_ds:
                        test_step(test_images, test_labels)
                    template = 'Epoch {}, Loss: {}, Test Loss: {}'
                    print(template.format(epoch + 1,
                                          trainer.train_loss.result(),
                                          validator.test_loss.result()))

                    loss_plotter.add("train", train_counter,
                                     tf.cast(trainer.train_loss.result(), tf.float64).numpy())

                    validator.test_loss.reset_states()

            # Reset the metrics for the next epoch
            trainer.train_loss.reset_states()

    finally:
        loss_plotter.plot()


def visualize_latent(ae, data, label, title, output_path, max_examples, embed_tech):
    categorical_plotter = CategoricalPlotter(np.unique(label), title, output_path)

    latent_vecs = ae.encode(data[:min(max_examples, data.shape[0])])
    if embed_tech == "lda":
        lda = LinearDiscriminantAnalysis(n_components=2)
        result = lda.fit_transform(latent_vecs, label[:min(max_examples, label.shape[0])])
    else:
        tsne = TSNE(n_components=2)
        result = tsne.fit_transform(latent_vecs)

    for i in range(result.shape[0]):
        categorical_plotter.add(label[i], result[i, 0], result[i, 1])

    categorical_plotter.plot()


def display_reconstruction(model, image, title, output_path):
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(image[:, :, 0])
    ax1.set_title("original")
    net_input = image[tf.newaxis, ...]
    ax2.imshow(model(net_input)[0, :, :, 0])
    ax2.set_title("prediction")

    fig.suptitle(title)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plt.savefig(os.path.join(output_path, title + ".png"))


if __name__ == '__main__':
    args = get_args()
    tf.keras.backend.set_floatx('float64')
    # batches = args.batches
    # epochs = args.epochs
    optimizer = get_optimizer(args.optimizer)
    loss, loss_with_latent = get_loss(args.loss, args.batches, args.reg_flag, args.reg_num)

    dataset_builder = get_dataset if args.dstype == "num" else get_denoise_dataset
    train_ds, test_ds, x_test, y_test = dataset_builder(args.batches, args.percent)
    print("dataset is ready")

    network = get_network(args.nntype)

    trainer = Trainer(network, optimizer, loss, loss_with_latent)
    validator = Validator(network, loss, loss_with_latent)

    train_main(args.epochs, train_ds, test_ds, trainer, validator, args.plot_freq, args.nntype,
               args.output_path)
    network.summary()

    params_title = "[method={},loss={},ds_name={}".format(args.embed_tech, args.loss, args.dstype)
    if args.dstype == "denoise":
        params_title += ",p={}".format(args.percent)
    params_title += "]"

    vis_title = "MNIST_claster_{}".format(params_title)
    im_title = "reconstruct_{}".format(params_title)

    visualize_latent(network, x_test, y_test, vis_title, args.output_path,
                     args.max_visualization, args.embed_tech)

    display_reconstruction(network, x_test[0], im_title, args.output_path)



